# ==============================================================================
# 목적 : PDF OCR 모듈
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-26
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import logging, re, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from app.common.config import load_config
from app.common.hash import sha256_bytes
from app.common.parser import get_value
from app.common.runtime import now_utc

from app.storage.postgres import PostgresWriter, PostgresConfig
from app.storage.minio import MinIOWriter, MinIOConfig
from app.storage.opensearch import OpenSearchWriter, OpenSearchConfig

from app.indexing.index_bodies import (
    build_pdf_chunks_v1_body,
    build_pdf_pages_staging_v1_body,
    build_pdf_images_v1_body,
    build_pdf_images_staging_v1_body,
    build_pdf_tables_v1_body,
    build_pdf_tables_staging_v1_body,
)
from app.parsing.table_extractor import build_table_text, build_row_text

from app.parsing.regex import RE_HTML_TABLE, RE_KIND_BBOX

_log = logging.getLogger(__name__)


def _as_bool(v: Any, default: bool = False) -> bool:
    """
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}: return True
    if s in {"0", "false", "no", "n", "off", ""}: return False
    return default


def _sha256_hex(data: bytes) -> str:
    """
    """
    v = sha256_bytes(data)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    return str(v)


def _pick_table_bbox(
    *,
    order: int,
    table_bboxes: Optional[List[Optional[List[int]]]] = None,
) -> Optional[List[int]]:
    """
    """
    if not table_bboxes:
        return None
    idx = int(order) - 1
    if idx < 0 or idx >= len(table_bboxes):
        return None
    bb = table_bboxes[idx]
    if not bb:
        return None
    try:
        if isinstance(bb, list) and len(bb) == 4:
            return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    except Exception:
        return None
    return None


def _chunks(lst: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [lst]
    return [lst[i : i + n] for i in range(0, len(lst), n)]


@dataclass
class VLMConfig:
    """
    """
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
    """
    """
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


def build_context(*, config_path: Optional[Path], source_type: str) -> IngestContext:
    """
    """
    cfg_path = config_path or Path("config/config.yaml")
    cfg = load_config(cfg_path)

    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk_size = int(get_value(cfg, "opensearch.bulk_size", 500))
    tables_enabled = _as_bool(get_value(cfg, "tables.enabled", True))

    prompt_ocr = str(get_value(cfg, "prompt_ocr", "")).strip()
    prompt_img_desc = str(get_value(cfg, "prompt_img_desc", "")).strip()

    vlm = VLMConfig(
        url=str(get_value(cfg, "vlm.url", "")).strip(),
        model=str(get_value(cfg, "vlm.model", "")).strip(),
        api_key=(str(get_value(cfg, "vlm.api_key", "")).strip() or None),
        timeout_sec=int(get_value(cfg, "vlm.timeout_sec", 3600)),
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

    text_index = str(get_value(cfg, "opensearch.text_index", "pdf_chunks_v1"))
    image_index = str(get_value(cfg, "opensearch.image_index", "pdf_images_v1"))
    table_index = str(get_value(cfg, "opensearch.table_index", "pdf_tables_v1"))
    pages_staging_index = str(get_value(cfg, "opensearch.pages_staging_index", "pdf_pages_staging_v1"))
    images_staging_index = str(get_value(cfg, "opensearch.images_staging_index", "pdf_images_staging_v1"))
    tables_staging_index = str(get_value(cfg, "opensearch.tables_staging_index", "pdf_tables_staging_v1"))

    os_text = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=text_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_image = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=image_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_table = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=table_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_pages_stage = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=pages_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_images_stage = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=images_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_tables_stage = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=tables_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))

    try:
        os_text.ensure_index(body=build_pdf_chunks_v1_body())
        os_image.ensure_index(body=build_pdf_images_v1_body())
        os_table.ensure_index(body=build_pdf_tables_v1_body())
        os_pages_stage.ensure_index(body=build_pdf_pages_staging_v1_body())
        os_images_stage.ensure_index(body=build_pdf_images_staging_v1_body())
        os_tables_stage.ensure_index(body=build_pdf_tables_staging_v1_body())
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


def _parse_html_table(html: str) -> Tuple[List[str], List[List[str]]]:
    """
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return [], []
    
    header: List[str] = []
    thead = table.find("thead")
    if thead:
        ths = thead.find_all(["th", "td"])
        header = [th.get_text(" ", strip=True) for th in ths]

    rows: List[List[str]] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        rows.append([td.get_text(" ", strip=True) for td in tds])

    if not header and rows:
        header = [f"col_{i+1}" for i in range(len(rows[0]))]

    if header:
        w = len(header)
        norm_rows: List[List[str]] = []
        for r in rows:
            if len(r) < w:
                r = r + [""] * (w - len(r))
            elif len(r) > w:
                r = r[:w]
            norm_rows.append(r)
        rows = norm_rows

    return header, rows


def _parse_plain_table_text(raw_text: str) -> Tuple[List[str], List[List[str]]]:
    """Plain text 테이블을 휴리스틱으로 파싱합니다.
    - 줄 단위 분리
    - 탭/2칸 이상 공백/파이프(|) 기준으로 열 분리
    """
    if not raw_text:
        return [], []
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    # Drop stray bbox tag line if it leaked into plain text.
    lines = [ln for ln in lines if not re.match(r"^table\[\[", ln, flags=re.IGNORECASE)]
    if not lines:
        return [], []

    def split_cols(line: str) -> List[str]:
        if "|" in line:
            return [c.strip() for c in line.split("|") if c.strip()]
        if "\t" in line:
            return [c.strip() for c in line.split("\t") if c.strip()]
        cols = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
        if len(cols) <= 1:
            return [line.strip()]
        return cols

    rows = [split_cols(ln) for ln in lines]
    if not rows:
        return [], []

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    w = len(header)
    norm = []
    for r in body:
        if len(r) < w:
            r = r + [""] * (w - len(r))
        elif len(r) > w:
            r = r[:w]
        norm.append(r)

    return header, norm


def stage_tables_from_text(
    *,
    ctx: IngestContext,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    viewer_uri: str,
    page_no: int,
    page_text: str,
    doc_sha256: str,
    table_bboxes: Optional[List[Optional[List[int]]]] = None,
) -> int:
    """
    """
    if not ctx.tables_enabled:
        return 0

    text = page_text or ""
    htmls = RE_HTML_TABLE.findall(text)

    def _extract_table_blocks(md: str) -> List[str]:
        blocks: List[str] = []
        if not md:
            return blocks

        matches = list(RE_KIND_BBOX.finditer(md))
        if not matches:
            return blocks

        for idx, m in enumerate(matches):
            if (m.group("kind") or "").lower() != "table":
                continue
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md)
            chunk = md[start:end]
            chunk = chunk.replace("\r", "").strip()
            if not chunk:
                continue
            # Normalize: drop leading newlines and whitespace-only lines.
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            raw_text = "\n".join(lines).strip()
            if raw_text:
                blocks.append(raw_text)

        return blocks

    staged = 0
    now = now_utc()
    os_docs: List[Dict[str, Any]] = []

    if htmls:
        for order, m in enumerate(htmls, start=1):
            raw_html = m[0] if isinstance(m, (tuple, list)) else m

            if raw_html.lstrip().lower().startswith("table[["):
                # If raw block already contains an actual HTML table, prefer that.
                inner_tables = RE_HTML_TABLE.findall(raw_html)
                if inner_tables:
                    raw_html = inner_tables[0] if not isinstance(inner_tables[0], (tuple, list)) else inner_tables[0][0]
                    header, rows = _parse_html_table(raw_html)
                    if not header:
                        raw_text = BeautifulSoup(raw_html, "html.parser").get_text("\n", strip=True)
                        header, rows = _parse_plain_table_text(raw_text)
                else:
                    raw_text = re.sub(r"^\s*table\[\[.*?\]\]\s*", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
                    raw_text = re.sub(r"<table\b[^>]*>|</table>", "", raw_text, flags=re.IGNORECASE)
                    raw_text = "\n".join([ln.strip() for ln in raw_text.splitlines() if ln.strip()])
                    if not raw_text:
                        continue
                    header, rows = _parse_plain_table_text(raw_text)
                if not header:
                    continue
                safe = "\n".join(["\t".join(r) for r in rows]) if rows else ""
                if not safe:
                    safe = raw_text if "raw_text" in locals() else ""
                safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                raw_html = "<table><tr><td>" + "<br/>".join(safe.splitlines()) + "</td></tr></table>"
            else:
                header, rows = _parse_html_table(raw_html)
                if not header:
                    raw_text = BeautifulSoup(raw_html, "html.parser").get_text("\n", strip=True)
                    header, rows = _parse_plain_table_text(raw_text)
                if not header:
                    continue

            _stage_table_doc(
                ctx=ctx,
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                viewer_uri=viewer_uri,
                page_no=page_no,
                doc_sha256=doc_sha256,
                table_bboxes=table_bboxes,
                order=order,
                raw_html=raw_html,
                header=header,
                rows=rows,
                now=now,
                os_docs=os_docs,
            )
            staged += 1
    else:
        blocks = _extract_table_blocks(text)
        if not blocks:
            return 0
        for order, raw_text in enumerate(blocks, start=1):
            safe = raw_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            raw_html = "<table><tr><td>" + "<br/>".join(safe.splitlines()) + "</td></tr></table>"
            header = ["content"]
            rows = [[raw_text]]
            _stage_table_doc(
                ctx=ctx,
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                viewer_uri=viewer_uri,
                page_no=page_no,
                doc_sha256=doc_sha256,
                table_bboxes=table_bboxes,
                order=order,
                raw_html=raw_html,
                header=header,
                rows=rows,
                now=now,
                os_docs=os_docs,
            )
            staged += 1

    if os_docs:
        ctx.os_tables_stage.bulk_upsert(os_docs, batch_size=ctx.bulk_size)

    return staged


def _stage_table_doc(
    *,
    ctx: IngestContext,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    viewer_uri: str,
    page_no: int,
    doc_sha256: str,
    table_bboxes: Optional[List[Optional[List[int]]]],
    order: int,
    raw_html: str,
    header: List[str],
    rows: List[List[str]],
    now,
    os_docs: List[Dict[str, Any]],
) -> None:
    table_id = f"{doc_id}:t{int(page_no):04d}:{int(order):04d}"
    table_sha = _sha256_hex(raw_html.encode("utf-8"))
    bbox = _pick_table_bbox(order=order, table_bboxes=table_bboxes)

    if ctx.pg is not None:
        with ctx.pg.cursor() as cur:
            PostgresWriter.upsert_pg_table(
                cur,
                table_id=table_id,
                doc_id=doc_sha256,
                page_no=int(page_no),
                order=int(order),
                bbox=bbox,
                row_count=int(len(rows)),
                col_count=int(len(header)),
                table_sha256=table_sha,
                raw_html=raw_html,
                header=header,
                rows=rows,
            )
        ctx.pg.commit()

    os_docs.append(
        {
            "_id": table_id,
            "_source": {
                "doc_id": doc_id,
                "doc_sha256": doc_sha256,
                "table_id": table_id,
                "doc_title": doc_title,
                "source_uri": source_uri,
                "viewer_uri": viewer_uri,
                "page_no": int(page_no),
                "order": int(order),
                "bbox": bbox,
                "header": header,
                "row_count": int(len(rows)),
                "col_count": int(len(header)),
                "raw_html": raw_html,
                "status": "pending",
                "attempts": 0,
                "last_error": "",
                "created_at": now,
                "updated_at": now,
            },
        }
    )


def finalize_tables_from_staging(*, ctx: IngestContext, doc_id: str, max_rows_embed: int = 500) -> Tuple[int, int]:
    """
    """
    query = {
        "query": {"bool": {"must": [{"term": {"doc_id": doc_id}}, {"term": {"status": "pending"}}]}},
        "sort": [{"page_no": "asc"}, {"order": "asc"}],
    }

    table_docs: List[Dict[str, Any]] = []
    row_docs: List[Dict[str, Any]] = []
    stage_updates: List[Dict[str, Any]] = []

    for hit in ctx.os_tables_stage.scan(query=query, size=500):
        src = hit.get("_source") or {}
        if not src:
            continue

        table_id = str(src.get("table_id") or hit.get("_id") or "")
        if not table_id:
            continue
        
        header = src.get("header") or []
        raw_html = str(src.get("raw_html") or "")
        page_no = int(src.get("page_no") or 0)
        order = int(src.get("order") or 0)
        bbox = src.get("bbox")
        doc_sha256 = src.get("doc_sha256")
        doc_title = src.get("doc_title") or ""
        source_uri = src.get("source_uri") or ""
        viewer_uri = src.get("viewer_uri") or ""
        source_type = src.get("source_type") or ""

        parsed_header, parsed_rows = _parse_html_table(raw_html) if raw_html else ([], [])
        if parsed_header:
            header = parsed_header
        rows = parsed_rows

        now = now_utc()

        table_text = build_table_text(header, rows, max_rows=max_rows_embed or 50) if header else ""
        table_docs.append(
            {
                "_id": table_id,
                "_source": {
                    "doc_id": doc_id,
                    "doc_sha256": doc_sha256,
                    "doc_title": doc_title,
                    "source_uri": source_uri,
                    "viewer_uri": viewer_uri,
                    "source_type": source_type,
                    "table_id": table_id,
                    "page_no": page_no,
                    "order": order,
                    "bbox": bbox,
                    "header": header,
                    "row_count": int(len(rows)),
                    "col_count": int(len(header) if header else 0),
                    "raw_html": raw_html,
                    "table_text": table_text,
                    "ingested_at": now,
                },
            }
        )

        if rows and max_rows_embed > 0:
            limit = min(len(rows), int(max_rows_embed))
            for ridx in range(limit):
                r = rows[ridx]
                row_id = f"{table_id}:r{ridx+1:04d}"
                row_obj, row_text = build_row_text(header, r)

                row_docs.append(
                    {
                        "_id": row_id,
                        "_source": {
                            "doc_id": doc_id,
                            "doc_sha256": doc_sha256,
                            "doc_title": doc_title,
                            "source_uri": source_uri,
                            "viewer_uri": viewer_uri,
                            "source_type": source_type,
                            "table_id": table_id,
                            "row_id": row_id,
                            "page_no": page_no,
                            "order": order,
                            "row_idx": int(ridx),
                            "bbox": bbox,
                            "header": header,
                            "row_obj_json": json.dumps(row_obj, ensure_ascii=False),
                            "row_text": row_text,
                            "ingested_at": now,
                        },
                    }
                )
        
        ctx.os_tables_stage.bulk_upsert(stage_updates, batch_size=ctx.bulk_size)

    if table_docs:
        # Use single index to surface errors deterministically.
        for d in table_docs:
            try:
                ctx.os_table._client.index(index=ctx.os_table.index, id=d["_id"], body=d["_source"], refresh=True)
            except Exception as e:
                _log.error("Table index failed: id=%s err=%s", d["_id"], e)
                raise

    if row_docs:
        try:
            for b in _chunks(row_docs, ctx.bulk_size):
                ctx.os_table.bulk_upsert(b, batch_size=ctx.bulk_size)
        except Exception as e:
            _log.warning("Row docs upsert failed (mapping may be strict). err=%s", e)

    if stage_updates:
        for b in _chunks(stage_updates, ctx.bulk_size):
            ctx.os_tables_stage.bulk_upsert(b, batch_size=ctx.bulk_size)
    
    return (len(table_docs), len(row_docs))


def process_images_from_staging(*, ctx: IngestContext, doc_id: str) -> int:
    """
    """
    query = {"query": {"bool": {"must": [{"term": {"doc_id": doc_id}}, {"term": {"status": "pending"}}]}}}

    updates: List[Dict[str, Any]] = []
    n = 0

    for hit in ctx.os_images_stage.scan(query=query, size=500):
        src = hit.get("_source") or {}
        if not src:
            continue
        stage_id = str(src.get("stage_id") or hit.get("_id") or "")
        if not stage_id:
            continue

        updates.append({"_id": stage_id, "_source": {**src, "status": "done", "updated_at": now_utc()}})
        n += 1

    if updates:
        for b in _chunks(updates, ctx.bulk_size):
            ctx.os_images_stage.bulk_upsert(b, batch_size=ctx.bulk_size,)

    return n


def index_chunks_from_md(
    *,
    ctx: IngestContext,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    viewer_uri: str,
    doc_sha256: str,
    md_text: str,
    write_pg: bool = False,
) -> Tuple[int, int]:
    """
    """
    parts = [p.strip() for p in (md_text or "").split("\n\n") if p.strip()]
    now = now_utc()

    docs: List[Dict[str, Any]] = []
    for i, text in enumerate(parts, start=1):
        chunk_id = f"{doc_id}:c{i:06d}"
        docs.append(
            {
                "_id": chunk_id,
                "_source": {
                    "doc_id": doc_id,
                    "doc_sha256": doc_sha256,
                    "chunk_id": chunk_id,
                    "doc_title": doc_title,
                    "source_uri": source_uri,
                    "viewer_uri": viewer_uri,
                    "order": i,
                    "text": text,
                    "created_at": now,
                    "updated_at": now,
                }
            }
        )

    if docs:
        ctx.os_text.bulk_upsert(docs, batch_size=ctx.bulk_size)
    
    return (len(docs), len(docs))
