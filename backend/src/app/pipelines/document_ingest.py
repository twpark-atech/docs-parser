import logging
import re
import base64
import json
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from app.common.config import load_config
from app.common.hash import sha256_bytes
from app.common.parser import get_value
from app.common.runtime import now_utc

from app.storage.postgres import PostgresConfig, PostgresWriter
from app.storage.minio import MinIOConfig, MinIOWriter
from app.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.storage.embedding import EmbeddingConfig, EmbeddingProvider
from app.parsing.regex import (
    RE_ANY_BBOX_CAPTURE,
    RE_IMAGE_ID_TOKEN,
    RE_SENT,
    RE_HTML_TR,
    RE_HTML_TD,
    RE_TABLE_KEY_PDF_META,
    RE_TABLE_KEY_DOCX_META,
)

from app.indexing.index_bodies import (
    build_chunks_body,
    build_pages_staging_body,
    build_images_body,
    build_images_staging_body,
    build_tables_body,
    build_tables_staging_body,
)

_log = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    url: str
    model: str
    api_key: Optional[str]
    timeout_sec: int

    prompt_ocr: str
    max_tokens: int
    temperature: float

    do_image_desc: bool
    prompt_img_desc: str
    img_desc_max_tokens: int
    img_desc_temperature: float


@dataclass
class IngestContext:
    cfg: Dict[str, Any]
    output_dir: Path
    bulk_size: int
    tables_enabled: bool

    vlm: VLMConfig

    minio_writer: MinIOWriter
    os_text: OpenSearchWriter
    os_image: OpenSearchWriter
    os_table: OpenSearchWriter
    os_pages_stage: OpenSearchWriter
    os_images_stage: OpenSearchWriter
    os_tables_stage: OpenSearchWriter

    pg: Optional[PostgresWriter] = None

    def close(self) -> None:
        try:
            if self.pg is not None:
                self.pg.close()
        except Exception:
            pass


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _resolve_vlm(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve active VLM settings.
    Priority:
    1) vlm.active_model + vlm.models.<name>
    2) legacy vlm.url/model/api_key/timeout_sec
    """
    vlm_models = get_value(cfg, "vlm.models", {}) or {}
    active_name = str(get_value(cfg, "vlm.active_model", "")).strip()

    selected: Dict[str, Any] = {}
    if isinstance(vlm_models, dict) and active_name:
        cand = vlm_models.get(active_name)
        if isinstance(cand, dict):
            selected = cand

    return {
        "name": active_name,
        "url": str(selected.get("url") or get_value(cfg, "vlm.url", "")).strip(),
        "model": str(selected.get("model") or get_value(cfg, "vlm.model", "")).strip(),
        "api_key": str(selected.get("api_key") or get_value(cfg, "vlm.api_key", "")).strip() or "EMPTY",
        "timeout_sec": int(selected.get("timeout_sec") or get_value(cfg, "vlm.timeout_sec", 3600)),
    }


def build_context(*, config_path: Optional[Path], source_type: str) -> IngestContext:
    cfg_path = config_path or Path("config/config.yml")
    cfg = load_config(cfg_path)

    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk_size = int(get_value(cfg, "opensearch.bulk_size", 500))
    tables_enabled = _as_bool(get_value(cfg, "tables.enabled", True))

    prompt_ocr = str(get_value(cfg, "prompt_ocr", "")).strip()
    prompt_img_desc = str(get_value(cfg, "prompt_img_desc", "")).strip()

    vlm_cfg = _resolve_vlm(cfg)
    vlm = VLMConfig(
        url=str(vlm_cfg.get("url") or "").strip(),
        model=str(vlm_cfg.get("model") or "").strip(),
        api_key=(str(vlm_cfg.get("api_key") or "").strip() or "EMPTY"),
        timeout_sec=int(vlm_cfg.get("timeout_sec") or 3600),
        prompt_ocr=prompt_ocr,
        max_tokens=int(get_value(cfg, "generation.max_tokens", 2048)),
        temperature=float(get_value(cfg, "generation.temperature", 0.0)),
        do_image_desc=_as_bool(get_value(cfg, "image_desc.enabled", False), default=False),
        prompt_img_desc=prompt_img_desc,
        img_desc_max_tokens=int(get_value(cfg, "image_desc.max_tokens", 256)),
        img_desc_temperature=float(get_value(cfg, "image_desc.temperature", 0.0)),
    )

    minio_cfg = MinIOConfig(
        endpoint=str(get_value(cfg, "minio.endpoint", "")),
        access_key=str(get_value(cfg, "minio.access_key", "")),
        secret_key=str(get_value(cfg, "minio.secret_key", "")),
        bucket=str(get_value(cfg, "minio.bucket", "")),
        secure=_as_bool(get_value(cfg, "minio.secure", "")),
    )
    minio_writer = MinIOWriter(minio_cfg)

    os_url = str(get_value(cfg, "opensearch.url", ""))
    if not os_url:
        raise ValueError("opensearch.url is required.")

    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = _as_bool(get_value(cfg, "opensearch.verify_certs", None))

    text_index = str(get_value(cfg, "opensearch.text_index", "chunks"))
    image_index = str(get_value(cfg, "opensearch.image_index", "images"))
    table_index = str(get_value(cfg, "opensearch.table_index", "tables"))
    pages_staging_index = str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging"))
    images_staging_index = str(get_value(cfg, "opensearch.images_staging_index", "images_staging"))
    tables_staging_index = str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging"))

    os_text = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=text_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    os_image = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=image_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    os_table = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=table_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    os_pages_stage = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=pages_staging_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    os_images_stage = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=images_staging_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    os_tables_stage = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=tables_staging_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )

    try:
        os_text.ensure_index(body=build_chunks_body())
        os_image.ensure_index(body=build_images_body())
        os_table.ensure_index(body=build_tables_body())
        os_pages_stage.ensure_index(body=build_pages_staging_body())
        os_images_stage.ensure_index(body=build_images_staging_body())
        os_tables_stage.ensure_index(body=build_tables_staging_body())
    except Exception as e:
        _log.warning("OpenSearch ensure_index skipped/failed (maybe managed by templates.) err=%s", e)

    pg_enabled = bool(get_value(cfg, "postgres.enabled", True))
    pg: Optional[PostgresWriter] = None
    if pg_enabled:
        pg_cfg = PostgresConfig(
            dsn=str(get_value(cfg, "postgres.dsn", "")),
            connect_timeout_sec=int(get_value(cfg, "postgres.connect_timeout_sec", 10)),
        )
        pg = PostgresWriter(pg_cfg)
        # Schema bootstrap is handled by API-layer _new_pg(). Re-running DDL here can
        # block long-running parse flows under lock contention.
        if _as_bool(get_value(cfg, "postgres.ensure_schema_in_context", False), default=False):
            pg.ensure_schema()

    return IngestContext(
        cfg=cfg,
        output_dir=output_dir,
        bulk_size=bulk_size,
        tables_enabled=tables_enabled,
        vlm=vlm,
        minio_writer=minio_writer,
        os_text=os_text,
        os_image=os_image,
        os_table=os_table,
        os_pages_stage=os_pages_stage,
        os_images_stage=os_images_stage,
        os_tables_stage=os_tables_stage,
        pg=pg,
    )


def _clean_ocr_text(s: str) -> str:
    out = str(s or "")
    out = re.sub(
        r"\[\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\]",
        "",
        out,
    )
    out = re.sub(
        r"\b(?:text|sub_title|image_caption|table|image|list_item|table_caption)\b",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\[[^\]]+\]", " ", out)
    out = re.sub(r"\[[^\]]+\]", " ", out)
    out = re.sub(r"^#+\s*", "", out)
    out = re.sub(r"<[^>]+>", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _bbox_from_text(s: str) -> Optional[List[int]]:
    m = RE_ANY_BBOX_CAPTURE.search(s or "")
    if not m:
        return None
    return [
        int(round(float(m.group("x0")))),
        int(round(float(m.group("y0")))),
        int(round(float(m.group("x1")))),
        int(round(float(m.group("y1")))),
    ]


def embedding_text(result: Dict[str, Any]):
    if "pdf" in result:
        source_key = "pdf"
        meta_key = "pdf_meta"
        pages = result.get("pdf") or []
    elif "docx" in result:
        source_key = "docx"
        meta_key = "docx_meta"
        pages = result.get("docx") or []
    else:
        return

    if not isinstance(pages, list):
        result[f"{source_key}_paragraphs"] = []
        result[f"{source_key}_sentences"] = []
        return

    cfg = load_config(Path("config/config.yaml"))
    meta = result.get(meta_key) or {}
    doc_id = str(meta.get("doc_id") or "")
    title = str(meta.get("title") or "")

    paragraph_rows: List[Dict[str, Any]] = []
    sentence_rows: List[Dict[str, Any]] = []
    global_para_idx = 0
    doc_char_cursor = 0

    for page_idx, page_text in enumerate(pages, start=1):
        if not isinstance(page_text, str) or not page_text.strip():
            continue
        page_no = page_idx if source_key == "pdf" else 0
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", page_text) if p.strip()]
        if source_key == "docx":
            raw_paragraphs = [page_text.strip()]
        for raw_para in raw_paragraphs:
            para_text = _clean_ocr_text(raw_para)
            if not para_text:
                continue
            global_para_idx += 1
            para_bbox = _bbox_from_text(raw_para) if source_key == "pdf" else None
            para_start = doc_char_cursor
            para_end = para_start + len(para_text)
            doc_char_cursor = para_end + 2
            paragraph_key = (
                f"{doc_id}:p{page_no:04d}:para{global_para_idx:06d}"
                if doc_id
                else f"p{page_no:04d}:para{global_para_idx:06d}"
            )

            paragraph_rows.append(
                {
                    "paragraph_key": paragraph_key,
                    "page_no": page_no,
                    "bbox": para_bbox,
                    "char_start": para_start,
                    "char_end": para_end,
                    "text": para_text,
                    "paragraph_idx": global_para_idx,
                }
            )

            raw_sents = [s.strip() for s in RE_SENT.findall(raw_para) if s and s.strip()]
            clean_sents = [_clean_ocr_text(s) for s in raw_sents]
            clean_sents = [s for s in clean_sents if s]
            if not clean_sents:
                clean_sents = [para_text]
                raw_sents = [raw_para]

            para_image_ids = [m.group("image_id").strip() for m in RE_IMAGE_ID_TOKEN.finditer(raw_para) if m.group("image_id").strip()]
            local_cursor = 0
            for s_idx, sent_text in enumerate(clean_sents, start=1):
                loc = para_text.find(sent_text, local_cursor)
                if loc < 0:
                    loc = local_cursor
                sent_start = para_start + max(0, loc)
                sent_end = sent_start + len(sent_text)
                local_cursor = max(0, loc) + len(sent_text)

                raw_sent = raw_sents[s_idx - 1] if s_idx - 1 < len(raw_sents) else raw_para
                sent_bbox = (_bbox_from_text(raw_sent) or para_bbox) if source_key == "pdf" else None
                image_ids = [m.group("image_id").strip() for m in RE_IMAGE_ID_TOKEN.finditer(raw_sent) if m.group("image_id").strip()]
                if not image_ids:
                    image_ids = para_image_ids

                sentence_rows.append(
                    {
                        "page_no": page_no,
                        "paragraph_idx": global_para_idx,
                        "sentence_idx": s_idx,
                        "paragraph_key": paragraph_key,
                        "bbox": sent_bbox,
                        "char_start": sent_start,
                        "char_end": sent_end,
                        "text": sent_text,
                        "image_ids": image_ids,
                    }
                )

    base_url = str(get_value(cfg, "embed_text.base_url", "")).strip()
    model = str(get_value(cfg, "embed_text.model", "")).strip()
    if base_url and model and sentence_rows:
        emb = EmbeddingProvider(
            EmbeddingConfig(
                base_url=base_url,
                model=model,
                timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
                max_batch_size=int(get_value(cfg, "embed_text.max_batch_size", 8)),
                truncate=bool(get_value(cfg, "embed_text.truncate", True)),
            )
        )
        texts = [r["text"] for r in sentence_rows]
        try:
            vecs = emb.embed(texts)
        except Exception as e:
            _log.warning("embedding_text failed: %s", e)
            vecs = [[] for _ in sentence_rows]
        for i, r in enumerate(sentence_rows):
            r["embedding"] = vecs[i] if i < len(vecs) else []
    else:
        for r in sentence_rows:
            r["embedding"] = []

    result[f"{source_key}_paragraphs"] = paragraph_rows
    result[f"{source_key}_sentences"] = sentence_rows

    if not doc_id:
        return

    try:
        ctx = build_context(config_path=Path("config/config.yaml"), source_type=source_key)
        try:
            retry_item_ids: set[str] = set()
            attempted_pages: set[int] = set()
            failed_pages: set[int] = set()
            if ctx.pg is not None:
                try:
                    retry_item_ids = set(ctx.pg.list_ingest_error_item_ids(domain="text", doc_id=doc_id))
                except Exception as e:
                    _log.warning("text error id load failed: %s", e)

            if ctx.pg is not None:
                ctx.pg.upsert_document(
                    sha256_hex=doc_id,
                    title=title or f"{doc_id}.{source_key}",
                    source_uri=str(meta.get("source_uri") or ""),
                    viewer_uri=str(meta.get("viewer_uri") or ""),
                    mime_type=str(meta.get("mime_type") or ""),
                    size_bytes=int(meta.get("size_bytes") or 0),
                    minio_bucket=str(meta.get("minio_bucket") or ""),
                    minio_key=str(meta.get("minio_key") or ""),
                    minio_etag=str(meta.get("minio_etag") or ""),
                )
                paragraph_inputs = [
                    {
                        "paragraph_key": p["paragraph_key"],
                        "page_no": p["page_no"],
                        "bbox": p["bbox"],
                        "char_start": p["char_start"],
                        "char_end": p["char_end"],
                    }
                    for p in paragraph_rows
                ]
                para_map = ctx.pg.upsert_paragraphs(doc_id=doc_id, paragraphs=paragraph_inputs)
                for s in sentence_rows:
                    chunk_id = f"{doc_id}:p{s['page_no']:04d}:para{s['paragraph_idx']:06d}:s{s['sentence_idx']:04d}"
                    if retry_item_ids and chunk_id not in retry_item_ids:
                        continue
                    pid = para_map.get(s["paragraph_key"])
                    if pid is None:
                        continue
                    try:
                        ctx.pg.upsert_sentences(
                            doc_id=doc_id,
                            sentences=[
                                {
                                    "paragraph_id": pid,
                                    "sentence_idx": int(s["sentence_idx"]),
                                    "page_no": int(s["page_no"]),
                                    "char_start": int(s["char_start"]),
                                    "char_end": int(s["char_end"]),
                                }
                            ],
                        )
                    except Exception as e:
                        ctx.pg.upsert_ingest_error(
                            domain="text",
                            doc_id=doc_id,
                            item_id=chunk_id,
                            page_no=int(s["page_no"]),
                            position=f"paragraph={int(s['paragraph_idx'])},sentence={int(s['sentence_idx'])}",
                            reason=str(e),
                            payload={"text": s.get("text", "")},
                        )
                        continue

            ingested_at = now_utc()
            for s in sentence_rows:
                chunk_id = f"{doc_id}:p{s['page_no']:04d}:para{s['paragraph_idx']:06d}:s{s['sentence_idx']:04d}"
                if retry_item_ids and chunk_id not in retry_item_ids:
                    continue
                page_no = int(s["page_no"])
                attempted_pages.add(page_no)
                source = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": s["text"],
                    "image_ids": s.get("image_ids") or [],
                    "embedding_model": model,
                    "ingested_at": ingested_at,
                }
                emb_vec = s.get("embedding") or []
                if emb_vec:
                    source["embedding"] = emb_vec
                try:
                    ctx.os_text.bulk_upsert([{"_id": chunk_id, "_source": source}], batch_size=1)
                    if ctx.pg is not None:
                        ctx.pg.clear_ingest_error(domain="text", doc_id=doc_id, item_id=chunk_id)
                except Exception as e:
                    failed_pages.add(page_no)
                    if ctx.pg is not None:
                        ctx.pg.upsert_ingest_error(
                            domain="text",
                            doc_id=doc_id,
                            item_id=chunk_id,
                            page_no=int(s["page_no"]),
                            position=f"paragraph={int(s['paragraph_idx'])},sentence={int(s['sentence_idx'])}",
                            reason=str(e),
                            payload={"text": s.get("text", "")},
                        )
            # PDF는 chunks 적재 성공 후 pages_staging 문서를 정리합니다.
            if source_key == "pdf":
                staged_pages: set[int] = set()
                try:
                    q = {"query": {"term": {"doc_id": doc_id}}}
                    for hit in ctx.os_pages_stage.scan(query=q, size=500):
                        src = hit.get("_source", {}) if isinstance(hit, dict) else {}
                        if str(src.get("status") or "").lower() != "done":
                            continue
                        p = int(src.get("page_no") or 0)
                        if p > 0:
                            staged_pages.add(p)
                except Exception as e:
                    _log.warning("pages_staging scan failed for cleanup: doc_id=%s err=%s", doc_id, e)

                candidate_pages = sorted(staged_pages)
                for page_no in candidate_pages:
                    # 해당 페이지에서 이번 실행 중 실패가 있으면 유지
                    if page_no in failed_pages:
                        continue

                    has_page_error = False
                    if ctx.pg is not None:
                        try:
                            with ctx.pg.cursor() as cur:
                                cur.execute(
                                    """
                                    SELECT COUNT(*)
                                    FROM text_errors
                                    WHERE doc_id=%s
                                      AND page_no=%s;
                                    """,
                                    (doc_id, int(page_no)),
                                )
                                r = cur.fetchone()
                            has_page_error = int((r or [0])[0] or 0) > 0
                        except Exception as e:
                            _log.warning("text_errors page check failed: doc_id=%s page_no=%s err=%s", doc_id, page_no, e)
                            has_page_error = True
                    if has_page_error:
                        continue
                    try:
                        deleted = ctx.os_pages_stage.delete_by_id(_id=f"{doc_id}:p{int(page_no):04d}")
                        if not deleted:
                            _log.warning(
                                "pages_staging delete returned false: doc_id=%s page_no=%s",
                                doc_id,
                                page_no,
                            )
                    except Exception as e:
                        _log.warning("pages_staging delete failed: doc_id=%s page_no=%s err=%s", doc_id, page_no, e)
        finally:
            ctx.close()
    except Exception as e:
        _log.warning("Storage upsert skipped/failed: %s", e)


def embedding_table(result: Dict[str, Any]) -> None:
    """MinIO 테이블을 LLM -> tables_staging -> embedding -> tables 순서로 처리합니다.

    처리 규칙:
    1) LLM 결과가 정상(table_html)인 테이블만 tables_staging에 upsert 합니다.
    2) tables_staging에서 doc_id 기준으로 하나씩 읽어 임베딩/최종 적재를 수행합니다.
    3) 최종 적재 성공 건만 tables_staging에서 삭제합니다.
    4) 실패 건은 tables_staging에 유지되어 다음 실행에서 재처리됩니다.
    """
    meta = result.get("pdf_meta") or result.get("docx_meta") or {}
    minio_key = str(meta.get("minio_key") or "").strip()
    doc_id = str(meta.get("doc_id") or "").strip()
    if not minio_key or not doc_id:
        result["tables_from_minio"] = []
        result["tables_staging"] = []
        return

    cfg = load_config(Path("config/config.yaml"))
    source_type = "pdf" if "pdf" in result else "docx"

    minio_cfg = MinIOConfig(
        endpoint=str(get_value(cfg, "minio.endpoint", "")),
        access_key=str(get_value(cfg, "minio.access_key", "")),
        secret_key=str(get_value(cfg, "minio.secret_key", "")),
        bucket=str(get_value(cfg, "minio.bucket", "")),
        secure=_as_bool(get_value(cfg, "minio.secure", "")),
    )
    writer = MinIOWriter(minio_cfg)
    prefix = minio_key.rsplit("/", 1)[0] if "/" in minio_key else ""
    tables_prefix = f"{prefix}/tables/" if prefix else "tables/"

    tables_from_minio: List[Dict[str, Any]] = []
    try:
        keys = sorted(k for k in writer.list_keys(prefix=tables_prefix) if k.lower().endswith(".md"))
        for key in keys:
            try:
                raw = writer.get_object_bytes(object_key=key)
            except Exception as e:
                _log.warning("MinIO table read failed: key=%s err=%s", key, e)
                continue
            tables_from_minio.append({"object_key": key, "table_md": raw.decode("utf-8", errors="replace")})
    except Exception as e:
        _log.warning("MinIO table listing failed: prefix=%s err=%s", tables_prefix, e)
    result["tables_from_minio"] = tables_from_minio

    def _extract_meta_from_key(key: str) -> Dict[str, Any]:
        page_no = 0
        order = 0
        m_pdf = RE_TABLE_KEY_PDF_META.search(key or "")
        if m_pdf:
            page_no = int(m_pdf.group("page") or 0)
            order = int(m_pdf.group("ord") or 0)
        else:
            m_docx = RE_TABLE_KEY_DOCX_META.search(key or "")
            if m_docx:
                page_no = 0
                order = int(m_docx.group("ord") or 0)
        return {"page_no": page_no, "order": order}

    def _strip_html_text(s: str) -> str:
        t = re.sub(r"<br\s*/?>", "\n", str(s or ""), flags=re.IGNORECASE)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _parse_table_cells(table_html: str) -> Dict[str, Any]:
        rows: List[List[str]] = []
        for rm in RE_HTML_TR.finditer(table_html or ""):
            row_html = rm.group("row") or ""
            cells = [_strip_html_text(cm.group("cell") or "") for cm in RE_HTML_TD.finditer(row_html)]
            if cells:
                rows.append(cells)
        if not rows:
            only = _strip_html_text(table_html)
            rows = [[only]] if only else []
        header = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        col_count = max((len(r) for r in rows), default=0)
        return {"header": header, "rows": data_rows, "row_count": len(rows), "col_count": col_count}

    def _coerce_json_list(v: Any) -> List[Any]:
        if isinstance(v, list):
            return v
        s = str(v or "").strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    llm_url = str(get_value(cfg, "llm.url", "")).strip()
    llm_model = str(get_value(cfg, "llm.model", "")).strip()
    llm_api_key = str(get_value(cfg, "llm.api_key", "")).strip() or "EMPTY"
    llm_timeout = int(get_value(cfg, "llm.timeout_sec", 120))
    llm_max_tokens = int(get_value(cfg, "llm.max_tokens", 2048))
    llm_temperature = float(get_value(cfg, "llm.temperature", 0.0))
    prompt_template = str(
        get_value(
            cfg,
            "llm.prompt",
            (
                "아래 테이블 텍스트를 HTML 표로 변환해줘.\n"
                "요구사항:\n"
                "1. 출력은 반드시 하나의 <table>...</table>만 반환한다.\n"
                "2. 표 내부는 <tr>, <td> 태그만 사용한다.\n"
                "3. HTML 속성(class, style, border 등)은 절대 넣지 않는다.\n"
                "4. 여러 컬럼이 붙어 있는 경우 문맥상 합리적으로 분리한다.\n"
                "5. 텍스트에 없는 값을 추정해서 추가하지 않는다.\n"
                "6. 설명/근거/코드블록 없이 HTML만 출력한다.\n\n"
                "입력:\n{table}"
            ),
        )
    )

    try:
        ctx = build_context(config_path=Path("config/config.yaml"), source_type=source_type)
    except Exception as e:
        _log.warning("build_context failed in embedding_table: %s", e)
        result["tables_staging"] = []
        return

    staged_ids: List[str] = []
    try:
        retry_table_ids: set[str] = set()
        if ctx.pg is not None:
            try:
                retry_table_ids = set(ctx.pg.list_ingest_error_item_ids(domain="table", doc_id=doc_id))
            except Exception as e:
                _log.warning("table error id load failed: %s", e)

        # 1) LLM 결과를 tables_staging에 적재
        # retry 상태여도 staging 잔여물과 새 원본 모두 재처리 가능하도록 항상 수행합니다.
        if llm_url and llm_model:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=llm_url, api_key=llm_api_key)
            except Exception as e:
                _log.warning("openai import/client failed. skip llm staging: %s", e)
                client = None

            stage_docs: List[Dict[str, Any]] = []
            for tbl_idx, row in enumerate(tables_from_minio, start=1):
                table_md = str(row.get("table_md") or "")
                object_key = str(row.get("object_key") or "")
                if not table_md.strip() or client is None:
                    continue
                prompt = prompt_template.replace("{table}", table_md)
                table_html = ""
                try:
                    resp = client.chat.completions.create(
                        model=llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                        timeout=llm_timeout,
                    )
                    content = ""
                    if getattr(resp, "choices", None):
                        content = str(getattr(resp.choices[0].message, "content", "") or "")
                    content = content.strip()
                    content = re.sub(r"^```(?:html)?\s*", "", content, flags=re.IGNORECASE)
                    content = re.sub(r"\s*```$", "", content)
                    m = re.search(r"<table>.*?</table>", content, flags=re.DOTALL | re.IGNORECASE)
                    table_html = m.group(0).strip() if m else content
                except Exception as e:
                    _log.warning("table llm processing failed: key=%s err=%s", object_key, e)
                    table_html = ""

                if not table_html:
                    continue

                key_meta = _extract_meta_from_key(object_key)
                page_no = int(key_meta["page_no"])
                order = int(key_meta["order"])
                if order <= 0:
                    order = tbl_idx
                if source_type == "docx":
                    page_no = 0

                table_sha = sha256_bytes(table_html.encode("utf-8", errors="ignore"))
                table_id = f"{doc_id}:p{page_no:04d}:t{order:02d}:{table_sha[:12]}"
                parsed = _parse_table_cells(table_html)
                stage_id = table_id
                staged_ids.append(stage_id)
                now = now_utc()
                stage_docs.append(
                    {
                        "_id": stage_id,
                        "_source": {
                            "doc_id": doc_id,
                            "table_id": table_id,
                            "doc_title": str(meta.get("title") or ""),
                            "source_uri": str(meta.get("source_uri") or ""),
                            "pdf_uri": str(meta.get("viewer_uri") or ""),
                            "page_no": page_no,
                            "order": order,
                            "table_sha256": table_sha,
                            "raw_html": table_html,
                            "header_json": json.dumps(parsed["header"], ensure_ascii=False),
                            "rows_json": json.dumps(parsed["rows"], ensure_ascii=False),
                            "status": "ready",
                            "attempts": 1,
                            "last_error": "",
                            "created_at": now,
                            "updated_at": now,
                        },
                    }
                )
        if stage_docs:
            ctx.os_tables_stage.bulk_upsert(stage_docs, batch_size=ctx.bulk_size)
            # 바로 scan으로 소비하므로 refresh를 보장합니다.
            ctx.os_tables_stage.refresh()

        result["tables_staging"] = staged_ids

        # 2) staging에서 하나씩 읽어 임베딩/최종적재 성공 시 staging 삭제
        table_model = str(get_value(cfg, "embed_text.model", "")).strip()
        table_base_url = str(get_value(cfg, "embed_text.base_url", "")).strip()
        embedder: Optional[EmbeddingProvider] = None
        if table_model and table_base_url:
            embedder = EmbeddingProvider(
                EmbeddingConfig(
                    base_url=table_base_url,
                    model=table_model,
                    timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
                    max_batch_size=1,
                    truncate=bool(get_value(cfg, "embed_text.truncate", True)),
                )
            )

        query = {"query": {"term": {"doc_id": doc_id}}}
        staged_hits = list(ctx.os_tables_stage.scan(query=query, size=500))
        for hit in staged_hits:
            stage_id = str(hit.get("_id") or "")
            src = hit.get("_source", {}) if isinstance(hit, dict) else {}
            if str(src.get("status") or "").lower() not in {"ready", "failed", "pending", "running"}:
                continue
            try:
                src_running = dict(src)
                src_running["status"] = "running"
                src_running["attempts"] = int(src_running.get("attempts") or 0) + 1
                src_running["updated_at"] = now_utc()
                ctx.os_tables_stage.bulk_upsert([{"_id": stage_id, "_source": src_running}], batch_size=1)
                src = src_running

                table_id = str(src.get("table_id") or stage_id or "")
                if not table_id:
                    table_id = f"{doc_id}:table:{sha256_bytes(str(stage_id).encode('utf-8'))[:12]}"
                table_html = str(src.get("raw_html") or src.get("table_text") or "")
                page_no = int(src.get("page_no") or 0)
                order = int(src.get("order") or 0)
                table_sha = str(src.get("table_sha256") or sha256_bytes(table_html.encode("utf-8", errors="ignore")))
                header = _coerce_json_list(src.get("header_json"))
                rows = _coerce_json_list(src.get("rows_json"))
                if not isinstance(header, list):
                    header = []
                if not isinstance(rows, list):
                    rows = []
                # 이전 스테이징 스키마/깨진 JSON 대비 폴백
                if (not header and not rows) and table_html:
                    parsed_fb = _parse_table_cells(table_html)
                    header = parsed_fb.get("header") or []
                    rows = parsed_fb.get("rows") or []
                # rows를 2차원 문자열 배열로 정규화
                norm_rows: List[List[str]] = []
                for row in rows:
                    if isinstance(row, list):
                        norm_rows.append([str(c or "") for c in row])
                    else:
                        cell = str(row or "").strip()
                        if cell:
                            norm_rows.append([cell])
                rows = norm_rows
                row_count = len(rows) + (1 if header else 0)
                col_count = max([len(header)] + [len(r) for r in rows]) if (header or rows) else 0

                emb_vec: List[float] = []
                if embedder is not None and table_html:
                    emb_vec = embedder.embed([table_html])[0]

                os_source = {
                    "doc_id": doc_id,
                    "table_id": table_id,
                    "table_text": table_html,
                    "row_embedding_model": table_model,
                    "ingested_at": now_utc(),
                }
                if emb_vec:
                    os_source["row_embedding"] = emb_vec
                try:
                    ctx.os_table.bulk_upsert([{"_id": table_id, "_source": os_source}], batch_size=1)
                except Exception:
                    # 벡터 차원 불일치 등 임베딩 필드 오류 시 본문만 우선 적재
                    os_source.pop("row_embedding", None)
                    ctx.os_table.bulk_upsert([{"_id": table_id, "_source": os_source}], batch_size=1)

                if ctx.pg is not None:
                    ctx.pg.upsert_document(
                        sha256_hex=doc_id,
                        title=str(meta.get("title") or f"{doc_id}"),
                        source_uri=str(meta.get("source_uri") or ""),
                        viewer_uri=str(meta.get("viewer_uri") or ""),
                        mime_type=str(meta.get("mime_type") or ""),
                        size_bytes=int(meta.get("size_bytes") or 0),
                        minio_bucket=str(meta.get("minio_bucket") or ""),
                        minio_key=str(meta.get("minio_key") or ""),
                        minio_etag=str(meta.get("minio_etag") or ""),
                    )
                    with ctx.pg.cursor() as cur:
                        PostgresWriter.upsert_pg_table(
                            cur,
                            table_id=table_id,
                            doc_id=doc_id,
                            page_no=page_no,
                            order=order,
                            bbox=None,
                            row_count=row_count,
                            col_count=col_count,
                            table_sha256=table_sha,
                            raw_html=table_html,
                            header=header,
                            rows=rows,
                        )
                    ctx.pg.commit()
                    ctx.pg.clear_ingest_error(domain="table", doc_id=doc_id, item_id=table_id)

                # 성공 시 staging 삭제. 삭제 실패는 재처리/진단을 위해 오류로 전환합니다.
                deleted = ctx.os_tables_stage.delete_by_id(_id=stage_id)
                if not deleted:
                    raise RuntimeError(f"tables_staging delete failed: stage_id={stage_id}")
            except Exception as e:
                _log.warning("table finalizing failed; keep staging: stage_id=%s err=%s", stage_id, e)
                try:
                    src_failed = dict(src)
                    src_failed["status"] = "failed"
                    src_failed["last_error"] = str(e)
                    src_failed["updated_at"] = now_utc()
                    ctx.os_tables_stage.bulk_upsert([{"_id": stage_id, "_source": src_failed}], batch_size=1)
                except Exception:
                    pass
                if ctx.pg is not None:
                    try:
                        ctx.pg.upsert_ingest_error(
                            domain="table",
                            doc_id=doc_id,
                            item_id=str(src.get("table_id") or stage_id),
                            page_no=int(src.get("page_no") or 0),
                            position=f"order={int(src.get('order') or 0)}",
                            reason=str(e),
                            payload={"stage_id": stage_id},
                        )
                    except Exception:
                        pass
                continue
    finally:
        ctx.close()


def _recover_image_id_from_key(doc_id: str, object_key: str) -> str:
    name = object_key.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if stem.startswith(f"{doc_id}_"):
        rest = stem[len(doc_id) + 1 :]
        if rest.startswith("p") and "_img" in rest:
            return f"{doc_id}:{rest.replace('_', ':')}"
        if rest.startswith("img"):
            return f"{doc_id}:{rest}"
    if stem.startswith(doc_id):
        tail = stem[len(doc_id) :].lstrip("_")
        if tail:
            return f"{doc_id}:{tail.replace('_', ':')}"
    return f"{doc_id}:img{sha256_bytes(object_key.encode('utf-8'))[:16]}"


def _extract_page_no_from_image_id(image_id: str) -> int:
    m = re.search(r":p(?P<page>\d{1,6}):img", image_id)
    if not m:
        return 0
    return int(m.group("page") or 0)


def _embed_image_bytes(cfg: Dict[str, Any], image_bytes: bytes) -> List[float]:
    base_url = str(get_value(cfg, "embed_image.base_url", "")).strip()
    if not base_url:
        return []
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/embed"):
        endpoint = endpoint + "/embed"
    timeout = int(get_value(cfg, "embed_image.timeout_sec", 60))
    dimension = int(get_value(cfg, "embed_image.dimension", 0) or 0)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload: Dict[str, Any] = {"images_b64": [b64], "dimension": dimension}
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        _log.warning("image embedding request failed: %s", e)
        return []

    # Accept multiple common response shapes.
    if isinstance(data, dict):
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs:
            first = embs[0]
            if isinstance(first, list):
                return first
        emb = data.get("embedding")
        if isinstance(emb, list):
            return emb
        vec = data.get("vector")
        if isinstance(vec, list):
            return vec
        arr = data.get("data")
        if isinstance(arr, list) and arr:
            first = arr[0] if isinstance(arr[0], dict) else {}
            emb2 = first.get("embedding")
            if isinstance(emb2, list):
                return emb2
    return []


def embedding_image(result: Dict[str, Any]) -> None:
    meta = result.get("pdf_meta") or result.get("docx_meta") or {}
    minio_key = str(meta.get("minio_key") or "").strip()
    doc_id = str(meta.get("doc_id") or "").strip()
    if not minio_key or not doc_id:
        result["images_processed"] = []
        result["images_staging"] = []
        return

    cfg = load_config(Path("config/config.yaml"))
    source_type = "pdf" if "pdf" in result else "docx"
    minio_cfg = MinIOConfig(
        endpoint=str(get_value(cfg, "minio.endpoint", "")),
        access_key=str(get_value(cfg, "minio.access_key", "")),
        secret_key=str(get_value(cfg, "minio.secret_key", "")),
        bucket=str(get_value(cfg, "minio.bucket", "")),
        secure=_as_bool(get_value(cfg, "minio.secure", "")),
    )
    writer = MinIOWriter(minio_cfg)
    minio_bucket = str(get_value(cfg, "minio.bucket", ""))
    prefix = minio_key.rsplit("/", 1)[0] if "/" in minio_key else ""
    images_prefix = f"{prefix}/images/" if prefix else "images/"

    try:
        image_keys = sorted(
            k for k in writer.list_keys(prefix=images_prefix) if k.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bin"))
        )
    except Exception as e:
        _log.warning("MinIO image listing failed: prefix=%s err=%s", images_prefix, e)
        result["images_processed"] = []
        result["images_staging"] = []
        return

    try:
        ctx = build_context(config_path=Path("config/config.yaml"), source_type=source_type)
    except Exception as e:
        _log.warning("build_context failed in embedding_image: %s", e)
        result["images_processed"] = []
        result["images_staging"] = []
        return

    def _parse_minio_uri(uri: str) -> str:
        u = str(uri or "").strip()
        if u.startswith("minio://"):
            parts = u[len("minio://") :].split("/", 1)
            if len(parts) == 2:
                return parts[1]
        return u

    staged_ids: List[str] = []
    processed: List[Dict[str, Any]] = []
    try:
        retry_image_ids: set[str] = set()
        if ctx.pg is not None:
            try:
                retry_image_ids = set(ctx.pg.list_ingest_error_item_ids(domain="image", doc_id=doc_id))
            except Exception as e:
                _log.warning("image error id load failed: %s", e)

        # 1) description 생성 결과를 images_staging에 적재
        stage_docs: List[Dict[str, Any]] = []
        ord_map: Dict[int, int] = {}
        for key in image_keys:
            try:
                b = writer.get_object_bytes(object_key=key)
            except Exception as e:
                _log.warning("MinIO image read failed: key=%s err=%s", key, e)
                continue

            image_id = _recover_image_id_from_key(doc_id, key)
            if retry_image_ids and image_id not in retry_image_ids:
                continue
            page_no = _extract_page_no_from_image_id(image_id)
            ord_map[page_no] = ord_map.get(page_no, 0) + 1
            ord_no = ord_map[page_no]

            width = 0
            height = 0
            try:
                from PIL import Image
                from io import BytesIO

                with Image.open(BytesIO(b)) as im:
                    width, height = im.size
            except Exception:
                width, height = 0, 0

            img_sha = sha256_bytes(b)
            image_uri = f"minio://{minio_bucket}/{key}" if minio_bucket else key

            desc = ""
            if _as_bool(get_value(cfg, "image_desc.enabled", False), default=False):
                try:
                    from app.parsing.image_extractor import describe_image_short

                    desc = describe_image_short(
                        b,
                        vlm_url=str(ctx.vlm.url or ""),
                        vlm_model=str(ctx.vlm.model or ""),
                        vlm_api_key=str(ctx.vlm.api_key or ""),
                        timeout_sec=int(ctx.vlm.timeout_sec or 120),
                        prompt=str(get_value(cfg, "prompt_img_desc", "")),
                        max_tokens=int(get_value(cfg, "image_desc.max_tokens", 256)),
                        temperature=float(get_value(cfg, "image_desc.temperature", 0.0)),
                    )
                except Exception as e:
                    _log.warning("image description skipped/failed: key=%s err=%s", key, e)
                    desc = ""

            now = now_utc()
            stage_docs.append(
                {
                    "_id": image_id,
                    "_source": {
                        "doc_id": doc_id,
                        "image_id": image_id,
                        "doc_title": str(meta.get("title") or ""),
                        "source_uri": str(meta.get("source_uri") or ""),
                        "page_no": page_no,
                        "order": ord_no,
                        "pdf_uri": str(meta.get("viewer_uri") or ""),
                        "image_uri": image_uri,
                        "image_mime": "image/png",
                        "image_sha256": img_sha,
                        "width": int(width),
                        "height": int(height),
                        "bbox": None,
                        "desc_text": desc,
                        "status": "ready",
                        "attempts": 1,
                        "last_error": "",
                        "created_at": now,
                        "updated_at": now,
                    },
                }
            )
            staged_ids.append(image_id)
        if stage_docs:
            ctx.os_images_stage.bulk_upsert(stage_docs, batch_size=ctx.bulk_size)
            # 바로 scan으로 소비하므로 refresh를 보장합니다.
            ctx.os_images_stage.refresh()
        result["images_staging"] = staged_ids

        # 2) staging에서 하나씩 불러와 임베딩/최종적재, 성공 시 staging 삭제
        desc_embed_base = str(get_value(cfg, "embed_text.base_url", "")).strip()
        desc_embed_model = str(get_value(cfg, "embed_text.model", "")).strip()
        desc_embedder: Optional[EmbeddingProvider] = None
        if desc_embed_base and desc_embed_model:
            desc_embedder = EmbeddingProvider(
                EmbeddingConfig(
                    base_url=desc_embed_base,
                    model=desc_embed_model,
                    timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
                    max_batch_size=1,
                    truncate=bool(get_value(cfg, "embed_text.truncate", True)),
                )
            )

        query = {"query": {"term": {"doc_id": doc_id}}}
        staged_hits = list(ctx.os_images_stage.scan(query=query, size=500))
        for hit in staged_hits:
            stage_id = str(hit.get("_id") or "")
            src = hit.get("_source", {}) if isinstance(hit, dict) else {}
            if str(src.get("status") or "").lower() not in {"ready", "failed", "pending", "running"}:
                continue
            try:
                src_running = dict(src)
                src_running["status"] = "running"
                src_running["attempts"] = int(src_running.get("attempts") or 0) + 1
                src_running["updated_at"] = now_utc()
                ctx.os_images_stage.bulk_upsert([{"_id": stage_id, "_source": src_running}], batch_size=1)
                src = src_running

                image_id = str(src.get("image_id") or stage_id)
                page_no = int(src.get("page_no") or 0)
                ord_no = int(src.get("order") or 0)
                image_uri = str(src.get("image_uri") or "")
                image_sha = str(src.get("image_sha256") or "")
                width = int(src.get("width") or 0)
                height = int(src.get("height") or 0)
                desc = str(src.get("desc_text") or "")
                object_key = _parse_minio_uri(image_uri)
                try:
                    img_bytes = writer.get_object_bytes(object_key=object_key) if object_key else b""
                except Exception as e:
                    _log.warning("image bytes read failed; continue without image embedding: image_id=%s err=%s", image_id, e)
                    img_bytes = b""

                desc_vec: List[float] = []
                if desc and desc_embedder is not None:
                    try:
                        desc_vec = desc_embedder.embed([desc])[0]
                    except Exception as e:
                        _log.warning("desc embedding failed; continue: image_id=%s err=%s", image_id, e)
                        desc_vec = []
                img_vec = _embed_image_bytes(cfg, img_bytes) if img_bytes else []

                os_source = {
                    "doc_id": doc_id,
                    "image_id": image_id,
                    "desc_text": desc,
                    "desc_embedding_model": desc_embed_model,
                    "image_embedding_model": str(get_value(cfg, "embed_image.model", "")),
                    "ingested_at": now_utc(),
                }
                if desc_vec:
                    os_source["desc_embedding"] = desc_vec
                if img_vec:
                    os_source["image_embedding"] = img_vec
                try:
                    ctx.os_image.bulk_upsert([{"_id": image_id, "_source": os_source}], batch_size=1)
                except Exception:
                    # 벡터 차원/매핑 이슈가 있으면 텍스트 메타만 우선 적재
                    os_source.pop("desc_embedding", None)
                    os_source.pop("image_embedding", None)
                    ctx.os_image.bulk_upsert([{"_id": image_id, "_source": os_source}], batch_size=1)

                if ctx.pg is not None:
                    ctx.pg.upsert_document(
                        sha256_hex=doc_id,
                        title=str(meta.get("title") or f"{doc_id}"),
                        source_uri=str(meta.get("source_uri") or ""),
                        viewer_uri=str(meta.get("viewer_uri") or ""),
                        mime_type=str(meta.get("mime_type") or ""),
                        size_bytes=int(meta.get("size_bytes") or 0),
                        minio_bucket=str(meta.get("minio_bucket") or ""),
                        minio_key=str(meta.get("minio_key") or ""),
                        minio_etag=str(meta.get("minio_etag") or ""),
                    )
                    ctx.pg.upsert_doc_images(
                        doc_id=doc_id,
                        images=[
                            {
                                "image_id": image_id,
                                "page_no": page_no,
                                "order": ord_no,
                                "image_uri": image_uri,
                                "image_sha256": image_sha,
                                "width": width,
                                "height": height,
                                "bbox": None,
                                "crop_bbox": None,
                                "det_bbox": None,
                                "caption_bbox": None,
                                "caption": "",
                                "description": "",
                            }
                        ],
                    )
                    ctx.pg.clear_ingest_error(domain="image", doc_id=doc_id, item_id=image_id)

                deleted = ctx.os_images_stage.delete_by_id(_id=stage_id)
                if not deleted:
                    raise RuntimeError(f"images_staging delete failed: stage_id={stage_id}")

                processed.append(
                    {
                        "image_id": image_id,
                        "page_no": page_no,
                        "ord": ord_no,
                        "image_uri": image_uri,
                        "image_sha256": image_sha,
                        "width": width,
                        "height": height,
                        "description": desc,
                    }
                )
            except Exception as e:
                _log.warning("image finalizing failed; keep staging: stage_id=%s err=%s", stage_id, e)
                try:
                    src_failed = dict(src)
                    src_failed["status"] = "failed"
                    src_failed["last_error"] = str(e)
                    src_failed["updated_at"] = now_utc()
                    ctx.os_images_stage.bulk_upsert([{"_id": stage_id, "_source": src_failed}], batch_size=1)
                except Exception:
                    pass
                if ctx.pg is not None:
                    try:
                        ctx.pg.upsert_ingest_error(
                            domain="image",
                            doc_id=doc_id,
                            item_id=str(src.get("image_id") or stage_id),
                            page_no=int(src.get("page_no") or 0),
                            position=f"order={int(src.get('order') or 0)}",
                            reason=str(e),
                            payload={"stage_id": stage_id, "image_uri": str(src.get("image_uri") or "")},
                        )
                    except Exception:
                        pass
                continue
    finally:
        ctx.close()

    result["images_processed"] = processed
