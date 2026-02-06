# ==============================================================================
# 목적 : Postgres 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-26
# AI 활용 여부 : 
# ==============================================================================

from __future__ import annotations

import hashlib, logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import execute_values, Json

from app.common.runtime import now_utc

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostgresConfig:
    """PostgresWriter 연결 설정을 담는 설정 객체입니다.
    Ingest 파이프라인(pdf_ingest/common_ingest)에서 Postgres 연결에 필요한 DSN 및 기본 연결 옵션을 제공합니다.

    Attributes:
        dsn: psycopg2.connect에 전달할 DSN 문자열.
        connect_timeout_sec: 연결 타임아웃(초).
    """
    dsn: str
    connect_timeout_sec: int = 10


class PostgresWriter:
    """Ingest 파이프라인에서 사용하는 Postgres 저장 유틸리티입니다.
    문서 단위로 다음 데이터를 관리합니다.
    - documents: 문서 메타 upsert.
    - doc_progress / doc_page_holes: 페이지 처리 진행률 및 hole 기반 resume.
    - paragraphs / sentences: 문단/문장 단위 메타 upsert.
    - doc_tables / doc_table_cells: 테이블 및 셀 upsert.

    Attributes:
        cfg: Postgres 연결 설정(PostgresConfig).
        _conn: psycopg2 connection.
    """

    HOLE_STATUS_PENDING = "pending"
    HOLE_STATUS_RUNNING = "running"
    HOLE_STATUS_FAILED = "failed"

    def __init__(self, cfg: PostgresConfig):
        """PostgresWriter를 초기화하고 DB 연결을 생성합니다.

        Args:
            cfg: Postgres 연결 설정.
        """
        self.cfg = cfg
        self._conn = psycopg2.connect(dsn=cfg.dsn, connect_timeout=cfg.connect_timeout_sec)
        self._conn.autocommit = False

    def close(self) -> None:
        """DB 연결을 종료합니다.
        
        예외가 발생하더라도 외부로 전파하지 않고 무시합니다.
        """
        try:
            self._conn.close()
        except Exception:
            pass

    def commit(self) -> None:
        """현재 트랜잭션을 커밋합니다."""
        self._conn.commit()

    def rollback(self) -> None:
        """현재 트랜잭션을 롤백합니다.

        예외가 발생하더라도 외부로 전파하지 않고 무시합니다.
        """
        try:
            self._conn.rollback()
        except Exception:
            pass

    @contextmanager
    def cursor(self):
        """DB cursor 컨텍스트를 제공합니다.

        Yields:
            psycopg2 cursor 객체.
        """
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def ensure_schema(self) -> None:
        """Ingest에 필요한 최소 테이블/인덱스 스키마를 생성합니다.

        생성 대상:
            - documents
            - doc_progress
            - doc_page_holes
            - paragraphs
            - sentences
            - doc_tables
            - doc_table_cells

        Raises:
            psycopg2.Error: DDL 실행에 실패(권한/문법/연결 문제 등)할 경우.
        """
        with self.cursor() as cur:
            # documents
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    sha256 CHAR(64) PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_uri TEXT,
                    viewer_uri TEXT,
                    mime_type TEXT,
                    size_bytes BIGINT,
                    minio_bucket TEXT,
                    minio_key TEXT,
                    minio_etag TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            # progress
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_progress (
                    doc_id CHAR(64) PRIMARY KEY REFERENCES documents(sha256) ON DELETE CASCADE,
                    total_pages INT NOT NULL,
                    contiguous_done_until INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            # holes
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_page_holes (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id CHAR(64) NOT NULL REFERENCES documents(sha256) ON DELETE CASCADE,
                    page_no INT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INT NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (doc_id, page_no)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_page_holes_doc ON doc_page_holes(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_page_holes_doc_status ON doc_page_holes(doc_id, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_page_holes_doc_page ON doc_page_holes(doc_id, page_no);")
            # paragraph
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS paragraphs (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id CHAR(64) NOT NULL REFERENCES documents(sha256) ON DELETE CASCADE,
                    paragraph_key TEXT NOT NULL UNIQUE,
                    page_no INT NOT NULL,
                    bbox JSONB,
                    char_start INT NOT NULL,
                    char_end INT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_doc ON paragraphs(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_page ON paragraphs(doc_id, page_no);")
            # sentences
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sentences (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id CHAR(64) NOT NULL REFERENCES documents(sha256) ON DELETE CASCADE,
                    paragraph_id BIGINT NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
                    sentence_idx INT NOT NULL,
                    page_no INT NOT NULL,
                    char_start INT NOT NULL,
                    char_end INT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (paragraph_id, sentence_idx)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_doc ON sentences(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_para ON sentences(paragraph_id);")
            # tables
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_tables (
                    table_id TEXT PRIMARY KEY,
                    doc_id CHAR(64) NOT NULL REFERENCES documents(sha256) ON DELETE CASCADE,
                    page_no INT NOT NULL,
                    ord INT NOT NULL,
                    bbox JSONB,
                    row_count INT NOT NULL,
                    col_count INT NOT NULL,
                    table_sha256 CHAR(64) NOT NULL,
                    raw_html TEXT NOT NULL,
                    header JSONB,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_tables_doc ON doc_tables(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_tables_doc_page ON doc_tables(doc_id, page_no);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_table_cells (
                    id BIGSERIAL PRIMARY KEY,
                    table_id TEXT NOT NULL REFERENCES doc_tables(table_id) ON DELETE CASCADE,
                    row_idx INT NOT NULL,
                    col_idx INT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (table_id, row_idx, col_idx)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_table_cells_table ON doc_table_cells(table_id);")

            # images
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_images (
                    image_id TEXT PRIMARY KEY,
                    doc_id CHAR(64) NOT NULL REFERENCES documents(sha256) ON DELETE CASCADE,
                    page_no INT NOT NULL,
                    ord INT NOT NULL,
                    image_uri TEXT,
                    image_sha256 CHAR(64),
                    width INT,
                    height INT,
                    bbox JSONB,
                    crop_bbox JSONB,
                    det_bbox JSONB,
                    caption_bbox JSONB,
                    caption TEXT,
                    description TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_images_doc ON doc_images(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_images_doc_page ON doc_images(doc_id, page_no);")

        self._conn.commit()

    @staticmethod
    def _lock_key(doc_id: str) -> int:
        """doc_id를 기반으로 pg_advisory_lock에 사용할 signed bigint 키를 생성합니다.

        Postgres pg_advisory_lock는 int8(bigint)를 받습니다.
        sha256 digest 상위 8바이트는 signed=True로 해석하여 bigint 범위를 만족하도록 합니다.

        Args:
            doc_id: 문서 ID(보통 sha256 hex).

        Returns:
            signed 64-bit 정수 키.
        """
        h = hashlib.sha256(doc_id.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big", signed=True)
    
    def advisory_lock(self, *, doc_id: str) -> None:
        """문서 단위 동시 실행을 방지하기 위해 advisory lock을 획득합니다.

        Args:
            doc_id: 잠금 키 생성에 사용할 문서 ID.

        Raises:
            psycopg2.Error: DB 호출에 실패한 경우.
        """
        key = self._lock_key(doc_id)
        with self.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s::bigint);", (key,))
        self._conn.commit()

    def advisory_unlock(self, *, doc_id: str) -> None:
        """획득한 advisory lock을 해제합니다.
        
        Args:
            doc_id: 잠금 키 생성에 사용할 문서 ID.

        Raises:
            psycopg2.Error: DB 호출에 실패한 경우.
        """
        key = self._lock_key(doc_id)
        with self.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s::bigint);", (key,))
        self._conn.commit()

    def upsert_document(
        self,
        *,
        sha256_hex: str,
        title: str,
        source_uri: str,
        viewer_uri: Optional[str],
        mime_type: Optional[str],
        size_bytes: Optional[int],
        minio_bucket: Optional[str],
        minio_key: Optional[str],
        minio_etag: Optional[str],
    ) -> None:
        """documents 테이블에 문서 메타를 upsert합니다.

        sha256(문서 해시)를 PK로 사용하며, 동일 sha256이 존재하면 최신 메타로 갱신합니다.

        Args:
            sha256_hex: 문서 SHA-256 hex(CHAR(64) PK).
            title: 문서 제목.
            source_uri: 원본 소스 URI.
            viewer_uri: 뷰어 접근 URI.
            mime_type: 문서 MIME 타입.
            size_bytes: 파일 크기(bytes).
            minio_bucket: MinIO bucket.
            minio_key: MinIO object key.
            minio_etag: MinIO etag.

        Raises:
            psycopg2.Error: 쿼리 실행에 실패한 경우.
        """
        now = now_utc()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    sha256, title, source_uri, viewer_uri, mime_type, size_bytes,
                    minio_bucket, minio_key, minio_etag,
                    created_at, updated_at
                )
                VALUES (
                    %(sha256)s, %(title)s, %(source_uri)s, %(viewer_uri)s, %(mime_type)s, %(size_bytes)s,
                    %(minio_bucket)s, %(minio_key)s, %(minio_etag)s,
                    %(created_at)s, %(updated_at)s
                )
                ON CONFLICT (sha256) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_uri = EXCLUDED.source_uri,
                    viewer_uri = EXCLUDED.viewer_uri,
                    mime_type = EXCLUDED.mime_type,
                    size_bytes = EXCLUDED.size_bytes,
                    minio_bucket = EXCLUDED.minio_bucket,
                    minio_key = EXCLUDED.minio_key,
                    minio_etag = EXCLUDED.minio_etag,
                    updated_at = EXCLUDED.updated_at;
                """,
                {
                    "sha256": sha256_hex,
                    "title": title,
                    "source_uri": source_uri,
                    "viewer_uri": viewer_uri,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "minio_bucket": minio_bucket,
                    "minio_key": minio_key,
                    "minio_etag": minio_etag,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        self._conn.commit()

    def upsert_doc_progress(self, *, doc_id: str, total_pages: int) -> None:
        """문서의 진행률(doc_progress)를 upsert합니다.
        
        total_pages는 기존 값보다 작아지지 않도록 GREATEST로 유지합니다.
        contiguous_done_until은 초기 0에서 시작합니다.

        Args:
            doc_id: 문서 ID(documents.sha256 참조).
            total_pages: 전체 페이지 수.

        Raises:
            psycopg2.Error: 쿼리 실행에 실패한 경우.
        """
        now = now_utc()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO doc_progress (doc_id, total_pages, contiguous_done_until, created_at, updated_at)
                VALUES (%(doc_id)s, %(total_pages)s, 0, %(now)s, %(now)s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    total_pages = GREATEST(doc_progress.total_pages, EXCLUDED.total_pages),
                    updated_at = EXCLUDED.updated_at;
                """,
                {"doc_id": doc_id, "total_pages": int(total_pages), "now": now},
            )
        self._conn.commit()

    def upsert_page_hole(
        self,
        *,
        doc_id: str,
        page_no: int,
        status: str,
        attempts_inc: int = 0,
        last_error: Optional[str] = None,
    ) -> None:
        """특정 페이지의 hole 상태를 upsert합니다.

        동일 (doc_id, page_no)에 대해:
        - status는 최신 값으로 갱신.
        - attempts는 기존 attempts + attempts_inc로 누적.
        - last_error는 새 값이 있으면 덮어쓰고, 없으면 기존 값 유지.

        Args:
            doc_id: 문서 ID.
            page_no: 페이지 번호.
            status: 페이지 상태(pending/running/failed 등).
            attempts_inc: 시도 횟수 증가분(음수는 0으로 보정).
            last_error: 실패 사유.

        Raises:
            psycopg2.Error: 쿼리 실행에 실패한 경우.
        """
        now = now_utc()
        inc = int(max(int(attempts_inc or 0), 0))
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO doc_page_holes (
                    doc_id, page_no, status, attempts, last_error, created_at, updated_at
                )
                VALUES (%(doc_id)s, %(page_no)s, %(status)s, %(attempts)s, %(last_error)s, %(now)s, %(now)s)
                ON CONFLICT (doc_id, page_no) DO UPDATE SET
                    status = EXCLUDED.status,
                    attempts = doc_page_holes.attempts + EXCLUDED.attempts,
                    last_error = COALESCE(EXCLUDED.last_error, doc_page_holes.last_error),
                    updated_at = EXCLUDED.updated_at;
                """,
                {
                    "doc_id": doc_id,
                    "page_no": int(page_no),
                    "status": str(status),
                    "attempts": inc,
                    "last_error": last_error,
                    "now": now,
                },
            )
        self._conn.commit()

    def reset_running_pages(self, *, doc_id: str, to_status: str) -> int:
        """running 상태의 hole들을 일괄 변경합니다.

        pdf_ingest 재시작 시, 비정상 종료로 남아있는 running 상태를 peding/failed 등으로 되돌릴 때 사용합니다.

        Args:
            doc_id: 문서 ID.
            to_status: 변경할 상태 값.

        Returns:
            상태가 변경된 row 수.
        """
        now = now_utc()
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE doc_page_holes
                SET status=%s, updated_at=%s
                WHERE doc_id=%s AND status=%s;
                """,
                (to_status, now, doc_id, self.HOLE_STATUS_RUNNING),
            )
            n = cur.rowcount
        self._conn.commit()
        return int(n or 0)
    
    def get_next_resume_page(self, doc_id: str) -> int:
        """다음 처리 시작 페이지(resume page)를 결정합니다.

        우선순위:
            1) holes 중 가장 작은 page_no (status 무관: pending/failed/running 포함).
            2) contiguous_done_until + 1

        Args:
            doc_id: 문서 ID.

        Returns:
            다음 처리 시작 페이지 번호(최소 1).
        """
        with self.cursor() as cur:
            cur.execute(
                "SELECT contiguous_done_until, total_pages FROM doc_progress WHERE doc_id=%s",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return 1
            contiguous, _total_pages = int(row[0] or 0), int(row[1] or 0)

            cur.execute("SELECT MIN(page_no) FROM doc_page_holes WHERE doc_id=%s", (doc_id,))
            m = cur.fetchone()
            min_hole = m[0] if m else None

        base = contiguous + 1
        if min_hole is None:
            return max(1, base)
        return int(min(int(min_hole), max(1, base)))
    
    def mark_page_done(self, *, doc_id: str, page_no: int) -> None:
        """페이지 처리 성공 시 holes 제거 및 contiguous_done_until을 전진시킵니다.

        처리 단계:
            1) (doc_id, page_no) hole 레코드를 삭제합니다.
            2) doc_progress row를 FOR UPDATE로 잠급니다.
            3) contiguous_done_until 이후 페이지가 hole에 없으면 연속 구간을 전진합니다.

        Args:
            doc_id: 문서 ID.
            page_no: 완료 처리할 페이지 번호.
        """
        now = now_utc()
        with self.cursor() as cur:
            cur.execute("DELETE FROM doc_page_holes WHERE doc_id=%s AND page_no=%s;", (doc_id, int(page_no)))

            cur.execute(
                "SELECT total_pages, contiguous_done_until FROM doc_progress WHERE doc_id=%s FOR UPDATE;",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    """
                    INSERT INTO doc_progress(doc_id, total_pages, contiguous_done_until, created_at, updated_at)
                    VALUES (%s, %s, 0, %s, %s)
                    ON CONFLICT (doc_id) DO NOTHING;
                    """,
                    (doc_id, int(page_no), now, now),
                )
                cur.execute(
                    "SELECT total_pages, contiguous_done_until FROM doc_progress WHERE doc_id=%s FOR UPDATE;",
                    (doc_id,),
                )
                row = cur.fetchone()

            total_pages, contiguous = int(row[0] or 0), int(row[1] or 0)

            while contiguous < total_pages:
                next_p = contiguous + 1
                cur.execute(
                    "SELECT 1 FROM doc_page_holes WHERE doc_id=%s AND page_no=%s LIMIT 1;",
                    (doc_id, next_p),
                )
                exists = cur.fetchone()
                if exists:
                    break
                contiguous = next_p

            cur.execute(
                """
                UPDATE doc_progress
                SET contiguous_done_until=%s, updated_at=%s
                WHERE doc_id=%s;
                """,
                (contiguous, now, doc_id),
            )

        self._conn.commit()

    def delete_paragraphs_for_doc(self, *, doc_id: str) -> int:
        """문서의 paragraphs를 삭제합니다(재실행 시 오염 방지 목적).

        paragraphs 삭제 시 sentences는 FK(ON DELETE CASCADE)로 함께 삭제됩니다.

        Args:
            doc_id: 문서 ID.

        Returns:
            삭제된 paragraph row 수.
        """
        with self.cursor() as cur:
            cur.execute("DELETE FROM paragraphs WHERE doc_id=%s;", (doc_id,))
            n = cur.rowcount
        self._conn.commit()
        return int(n or 0)
    
    def upsert_paragraphs(self, *, doc_id: str, paragraphs: List[Dict[str, Any]]) -> Dict[str, int]:
        """문단(paragraphs)을 bulk upsert하고 paragraph_key->id 매핑을 반환합니다.

        Args:
            doc_id: 문서 ID.
            paragraphs: 문단 리스트.
                각 원소는 최소 다음 키를 포함해야 합니다.
                - paragraph_key (str): 전역 유니크 키.
                - page_no (int)
                - bbox (dict|None)
                - char_start (int)
                - char_end (int)

        Returns:
            paragraph_key -> paragraph_id 매핑 딕셔너리.

        Raises:
            KeyError: paragraphs 항목에 paragraph_key가 없는 경우.
            psycopg2.Error: 쿼리 실행에 실패한 경우. 
        """
        if not paragraphs:
            return {}
        
        now = now_utc()
        rows = []
        for p in paragraphs:
            rows.append(
                (
                    doc_id,
                    str(p["paragraph_key"]),
                    int(p.get("page_no") or 0),
                    Json(p.get("bbox")) if p.get("bbox") is not None else None,
                    int(p.get("char_start") or 0),
                    int(p.get("char_end") or 0),
                    now,
                    now,
                )
            )
        
        sql = """
        INSERT INTO paragraphs (
            doc_id, paragraph_key, page_no, bbox, char_start, char_end, created_at, updated_at
        )
        VALUES %s
        ON CONFLICT (paragraph_key) DO UPDATE SET
            page_no = EXCLUDED.page_no,
            bbox = EXCLUDED.bbox,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            updated_at = EXCLUDED.updated_at
        RETURNING id, paragraph_key;
        """

        out: Dict[str, int] = {}
        with self.cursor() as cur:
            execute_values(cur, sql, rows, page_size=500)
            for pid, pkey in cur.fetchall():
                out[str(pkey)] = int(pid)
        self._conn.commit()
        return out
    
    def upsert_sentences(self, *, doc_id: str, sentences: List[Dict[str, Any]]) -> None:
        """문장(sentences)를 bulk upsert합니다.

        (paragraph_id, sentence_idx)가 UNIQUE이며, 동일 키가 존재하면 위치/offset 정보를 갱신합니다.

        Args:
            doc_id: 문서 ID.
            sentences: 문장 리스트.
                각 원소는 최소 다음 키를 포함해야 합니다.
                - paragraph_id (int)
                - sentences_idx (int)
                - page_no (int)
                - char_start (int)
                - char_end (int)

        Raises:
            KeyError: sentences 항목에 paragraph_id가 없는 경우.
            psycopg2.Error: 쿼리 실행에 실패한 경우.
        """
        if not sentences:
            return
        now = now_utc()
        rows = []
        for s in sentences:
            rows.append(
                (
                    doc_id,
                    int(s["paragraph_id"]),
                    int(s.get("sentence_idx") or 0),
                    int(s.get("page_no") or 0),
                    int(s.get("char_start") or 0),
                    int(s.get("char_end") or 0),
                    now,
                    now,
                )
            )

        sql = """
        INSERT INTO sentences (
            doc_id, paragraph_id, sentence_idx, page_no, char_start, char_end, created_at, updated_at
        )
        VALUES %s
        ON CONFLICT (paragraph_id, sentence_idx) DO UPDATE SET
            page_no = EXCLUDED.page_no,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            updated_at = EXCLUDED.updated_at;
        """

        with self.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        self._conn.commit()

    def upsert_doc_images(self, *, doc_id: str, images: List[Dict[str, Any]]) -> None:
        """doc_images에 이미지 메타데이터를 upsert합니다.

        Args:
            doc_id: 문서 ID(sha256).
            images: 이미지 메타 리스트.
        """
        if not images:
            return
        now = now_utc()
        rows = []
        for it in images:
            rows.append(
                (
                    str(it.get("image_id") or ""),
                    doc_id,
                    int(it.get("page_no") or 0),
                    int(it.get("order") or 0),
                    str(it.get("image_uri") or ""),
                    str(it.get("image_sha256") or ""),
                    int(it.get("width") or 0),
                    int(it.get("height") or 0),
                    Json(it.get("bbox")) if it.get("bbox") is not None else None,
                    Json(it.get("crop_bbox")) if it.get("crop_bbox") is not None else None,
                    Json(it.get("det_bbox")) if it.get("det_bbox") is not None else None,
                    Json(it.get("caption_bbox")) if it.get("caption_bbox") is not None else None,
                    str(it.get("caption") or ""),
                    str(it.get("description") or ""),
                    now,
                    now,
                )
            )

        sql = """
        INSERT INTO doc_images (
            image_id, doc_id, page_no, ord, image_uri, image_sha256, width, height,
            bbox, crop_bbox, det_bbox, caption_bbox, caption, description, created_at, updated_at
        )
        VALUES %s
        ON CONFLICT (image_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            page_no = EXCLUDED.page_no,
            ord = EXCLUDED.ord,
            image_uri = EXCLUDED.image_uri,
            image_sha256 = EXCLUDED.image_sha256,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            bbox = EXCLUDED.bbox,
            crop_bbox = EXCLUDED.crop_bbox,
            det_bbox = EXCLUDED.det_bbox,
            caption_bbox = EXCLUDED.caption_bbox,
            caption = EXCLUDED.caption,
            description = EXCLUDED.description,
            updated_at = EXCLUDED.updated_at;
        """

        with self.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        self._conn.commit()

    @staticmethod
    def upsert_pg_table(
        cur,
        *,
        table_id: str,
        doc_id: str,
        page_no: int,
        order: int,
        bbox: Optional[Any],
        row_count: int,
        col_count: int,
        table_sha256: str,
        raw_html: str,
        header: List[str],
        rows: List[List[str]],
    ) -> None:
        """테이블(doc_tables)과 셀(doc_table_cells)을 동일 트랜잭션 내에서 upsert합니다.

        common_ingest.stage_tables_from_text에서 호출되는 최소 계약 API입니다.
        외부에서 전달된 cursor(cur)를 사용하며, 본 메서드는 commit/rollback을 수행하지 않습니다.

        Args:
            cur: psycopg2 cursor(외부 트랜잭션 컨텍스트).
            table_id: 테이블 고유 ID(PK).
            doc_id: 문서 ID.
            page_no: 페이지 번호.
            order: 페이지 내 테이블 순서(ord).
            bbox: 테이블 bbox(jsonb로 저장) 또는 None.
            row_count: 행 수.
            col_count: 열 수.
            table_sha256: 테이블 원문 기반 해시.
            raw_html: 테이블 HTML 원문.
            header: 헤더 리스트(JSON 저장).
            rows: 셀 텍스트 2차원 배열.
        """
        now = now_utc()
        cur.execute(
            """
            INSERT INTO doc_tables(
                table_id, doc_id, page_no, ord, bbox, row_count, col_count, table_sha256,
                raw_html, header, created_at, updated_at
            )
            VALUES (
                %(table_id)s, %(doc_id)s, %(page_no)s, %(ord)s, %(bbox)s, %(row_count)s, %(col_count)s, %(table_sha256)s,
                %(raw_html)s, %(header)s, %(now)s, %(now)s            
            )
            ON CONFLICT (table_id) DO UPDATE SET
                page_no = EXCLUDED.page_no,
                ord = EXCLUDED.ord,
                bbox = EXCLUDED.bbox,
                row_count = EXCLUDED.row_count,
                col_count = EXCLUDED.col_count,
                table_sha256 = EXCLUDED.table_sha256,
                raw_html = EXCLUDED.raw_html,
                header = EXCLUDED.header,
                updated_at = EXCLUDED.updated_at;
            """,
            {
                "table_id": table_id, 
                "doc_id": doc_id, 
                "page_no": int(page_no), 
                "ord": int(order), 
                "bbox": Json(bbox) if bbox is not None else None, 
                "row_count": int(row_count), 
                "col_count": int(col_count), 
                "table_sha256": str(table_sha256),
                "raw_html": str(raw_html), 
                "header": Json(header),
                "now": now,
            },
        )

        cell_rows = []
        for r_idx, row in enumerate(rows):
            for c_idx, txt in enumerate(row):
                cell_rows.append((table_id, int(r_idx), int(c_idx), str(txt), now, now))

        if cell_rows:
            execute_values(
                cur,
                """
                INSERT INTO doc_table_cells(table_id, row_idx, col_idx, text, created_at, updated_at)
                VALUES %s
                ON CONFLICT (table_id, row_idx, col_idx) DO UPDATE SET
                    text = EXCLUDED.text,
                    updated_at = EXCLUDED.updated_at;
                """,
                cell_rows,
                page_size=2000,
            )

    def get_process_summary(self, *, doc_id: str) -> Dict[str, Any]:
        """
        """
        out: Dict[str, Any] = {"total_pages": 0, "done": 0, "failed": 0, "pending": 0, "running": 0}

        with self.cursor() as cur:
            cur.execute(
                "SELECT total_pages, contiguous_done_until FROM doc_progress WHERE doc_id=%s LIMIT 1;",
                (doc_id,),
            )
            r = cur.fetchone()
            if r:
                out["total_pages"] = int(r[0] or 0)
                out["done"] = int(r[1] or 0)

            cur.execute(
                """
                SELECT status, COUNT(*)
                FROM doc_page_holes
                WHERE doc_id=%s
                GROUP BY status;
                """,
                (doc_id,),
            )
            for st, cnt in cur.fetchall() or []:
                st = str(st)
                out[st] = int(cnt or 0)

        out["failed"] = int(out.get(self.HOLE_STATUS_FAILED) or 0)
        out["pending"] = int(out.get(self.HOLE_STATUS_PENDING) or 0)
        out["running"] = int(out.get(self.HOLE_STATUS_RUNNING) or 0)
        return out
