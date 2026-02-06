# ==============================================================================
# 목적 : PDF OCR 모듈
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-26
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import inspect, json, logging, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from app.common.hash import sha256_file, sha256_bytes
from app.common.ids import s3_uri, parse_s3_uri
from app.common.parser import get_value, pdf_to_page_pngs
from app.common.runtime import now_utc

from app.parsing.image_extractor import (
    extract_and_store_images_from_page,
    get_image_size_from_bytes,
    rewrite_image_det_bbox_to_image_id,
)
from app.parsing.ocr import ocr_page
from app.parsing.pdf import coerce_page_no_and_payload, materialize_png_payload
from app.parsing.regex import (
    RE_SENT,
    RE_HTML_TAG,
    RE_PIPE_TAG,
    RE_ANY_BBOX_CAPTURE,
    RE_REFDET_BLOCK,
    RE_KIND_BBOX,
)

from app.storage.embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider
from app.storage.minio import MinIOConfig, MinIOReader

from app.indexing.embedding import (
    ImageEmbedConfig,
    embed_images_bytes_batch,
    embed_texts,
)

from app.workflows.common_ingest import (
    build_context,
    IngestContext,
    stage_tables_from_text,
    finalize_tables_from_staging,
    index_chunks_from_md,
)

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
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _atomic_write_text(path: Path, text: str) -> None:
    """
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _build_md_with_real_page_no(pages: List[Tuple[int, str]]) -> str:
    """
    """
    out: List[str] = []
    for pn, txt in sorted(pages, key=lambda x: x[0]):
        out.append(f"## Page {pn}\n\n{(txt or '').strip()}\n\n---\n")
    return ("\n".join(out).strip() + "\n") if out else ""


def _load_pages_clean_from_text_dir(pages_text_dir: Path) -> List[Tuple[int, str]]:
    """
    """
    pages: List[Tuple[int, str]] = []
    if not pages_text_dir.exists():
        return pages
    for p in sorted(pages_text_dir.glob("page_*.clean.txt")):
        m = re.search(r"page_(\d{4})\.clean\.txt$", p.name)
        if not m:
            continue
        pn = int(m.group(1))
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            pages.append((pn, txt))
    pages.sort(key=lambda x: x[0])
    return pages


def _dump_page_items_checkpoint(pages_text_dir: Path, page_no: int, items: List[Dict[str, Any]]) -> None:
    """
    """
    p = pages_text_dir / f"page_{page_no:04d}.items.json"
    _atomic_write_text(p, json.dumps(items, ensure_ascii=False, indent=2))


def _parse_ref_det_items(text: str) -> List[Dict[str, Any]]:
    """
    """
    items: List[Dict[str, Any]] = []
    if not text:
        return items
    
    for m in RE_REFDET_BLOCK.finditer(text):
        ref = (m.group("ref") or "").strip()
        det = (m.group("det") or "").strip()

        bbox = None
        mb = RE_ANY_BBOX_CAPTURE.search(det)
        if mb:
            try:
                bbox = [
                    int(float(mb.group("x0"))),
                    int(float(mb.group("y0"))),
                    int(float(mb.group("x1"))),
                    int(float(mb.group("y1"))),
                ]
            except Exception:
                bbox = None

        items.append({"ref": ref, "det": det, "bbox": bbox, "span": (int(m.start()), int(m.end()))})

    if items:
        return items

    for m in RE_KIND_BBOX.finditer(text):
        ref = (m.group("kind") or "").strip().lower()
        try:
            bbox = [
                int(float(m.group("x0"))),
                int(float(m.group("y0"))),
                int(float(m.group("x1"))),
                int(float(m.group("y1"))),
            ]
        except Exception:
            bbox = None
        det = f"[[{m.group('x0')}, {m.group('y0')}, {m.group('x1')}, {m.group('y1')}]]"
        items.append({"ref": ref, "det": det, "bbox": bbox, "span": (int(m.start()), int(m.end()))})
    return items


def _build_image_caption_bbox_map(items: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    """
    out: Dict[str, List[int]] = {}
    n = len(items)

    for i, it in enumerate(items):
        if (it.get("ref") or "").lower() != "image":
            continue
        image_id = (it.get("det") or "").strip()
        if not image_id:
            continue

        for j in range(i + 1, min(i + 6, n)):
            jt = items[j]
            r = (jt.get("ref") or "").lower()
            if r == "image_caption" and jt.get("bbox"):
                out[image_id] = jt["bbox"]
                break
            if r == "image":
                break

    return out


def _collect_table_bboxes(items: List[Dict[str, Any]]) -> List[Optional[List[int]]]:
    """
    """
    bboxes: List[Optional[List[int]]] = []
    for it in items:
        if (it.get("ref") or "").lower() == "table":
            bboxes.append(it.get("bbox"))
    return bboxes


def _strip_tags_keep_det_payload(text: str) -> str:
    """
    """
    if not text:
        return ""
    
    # Remove image_id tokens and bbox tag lines early.
    text = re.sub(r"\b[a-f0-9]{64}:p\d+:img[0-9a-f]{16}\b", "", text, flags=re.IGNORECASE)
    text = RE_KIND_BBOX.sub("", text)

    def _det_repl(m: re.Match) -> str:
        gd = m.groupdict() or {}
        payload = (gd.get("det") or (m.group(1) if m.lastindex and m.lastindex >= 1 else "") or "").strip()
        if not payload:
            return ""
        if RE_ANY_BBOX_CAPTURE.fullmatch(payload.strip()):
            return ""
        return payload
    
    t = RE_REFDET_BLOCK.sub(_det_repl, text)
    t = RE_PIPE_TAG.sub("", t)
    t = RE_HTML_TAG.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _extract_first_bbox_from_text(text: str) -> Optional[List[int]]:
    """
    """
    if not text:
        return None
    m = RE_ANY_BBOX_CAPTURE.search(text)
    if not m:
        return None
    try:
        return [
            int(float(m.group("x0"))),
            int(float(m.group("y0"))),
            int(float(m.group("x1"))),
            int(float(m.group("y1"))),
        ]
    except Exception:
        return None
    

def _split_sentences_keep_indices(text: str) -> List[Dict[str, Any]]:
    """
    """
    t = (text or "")
    if not t.strip():
        return []
    
    out: List[Dict[str, Any]] = []
    for m in RE_SENT.finditer(t):
        s = m.group(0)
        if not s or not s.strip():
            continue
        out.append({"text": s, "char_start": m.start(), "char_end": m.end()})
    return out


def _extract_paragraphs_from_page_text(*, page_text_raw: str, page_no: int) -> List[Dict[str, Any]]:
    """
    """
    lines = (page_text_raw or "").splitlines()

    paras: List[Dict[str, Any]] = []
    buf: List[str] = []
    para_idx = 0

    def flush():
        nonlocal para_idx, buf
        raw_para = "\n".join([x for x in buf if x.strip()]).strip()
        if not raw_para:
            buf = []
            return
        
        bbox = _extract_first_bbox_from_text(raw_para)
        clean = _strip_tags_keep_det_payload(raw_para)
        if clean:
            paras.append({"para_idx": int(para_idx), "page_no": int(page_no), "bbox": bbox, "text": clean})
            para_idx += 1
        buf = []

    for ln in lines:
        if not ln.strip():
            flush()
            continue
        buf.append(ln)

    flush()
    return paras


def _upsert_pg_paragraphs_sentences_from_pages(
    *,
    ctx: IngestContext,
    doc_sha256: str,
    pages_raw: List[Tuple[int, str]],
) -> None:
    """
    """
    if ctx.pg is None:
        return
    
    para_rows: List[Dict[str, Any]] = []
    extracted_paras: List[Dict[str, Any]] = []

    for page_no, page_text_raw in pages_raw:
        paras = _extract_paragraphs_from_page_text(page_text_raw=page_text_raw, page_no=int(page_no))
        for p in paras:
            paragraph_key = f"{doc_sha256}:p{int(page_no):04d}:para{int(p['para_idx']):04d}"
            txt = p.get("text") or ""
            extracted_paras.append({**p, "paragraph_key": paragraph_key})

            para_rows.append(
                {
                    "paragraph_key": paragraph_key,
                    "page_no": int(page_no),
                    "bbox": p.get("bbox"),
                    "char_start": 0,
                    "char_end": len(txt),
                }
            )

    para_id_by_key = ctx.pg.upsert_paragraphs(doc_id=doc_sha256, paragraphs=para_rows)

    sent_rows: List[Dict[str, Any]] = []
    for p in extracted_paras:
        pid = para_id_by_key.get(p["paragraph_key"])
        if not pid:
            continue

        txt = p.get("text") or ""
        for s_idx, s in enumerate(_split_sentences_keep_indices(txt)):
            sent_rows.append(
                {
                    "paragraph_id": int(pid),
                    "sentence_idx": int(s_idx),
                    "page_no": int(p.get("page_no") or 0),
                    "char_start": int(s.get("char_start") or 0),
                    "char_end": int(s.get("char_end") or 0),
                }
            )

    ctx.pg.upsert_sentences(doc_id=doc_sha256, sentences=sent_rows)
    ctx.pg.commit()


def _pg_init_progress_if_needed(*, ctx: IngestContext, pg_doc_id: str, total_pages: int) -> None:
    """
    """
    if ctx.pg is None:
        return
    
    try:
        n = ctx.pg.reset_running_pages(doc_id=pg_doc_id, to_status=ctx.pg.HOLE_STATUS_PENDING)
        if n:
            _log.info("Reset running holes -> pending. doc_id=%s count=%d", pg_doc_id, n)
    except Exception as e:
        _log.info("reset_running_pages(holes) failed: doc_id=%s err=%s", pg_doc_id, e)

    try:
        ctx.pg.upsert_doc_progress(doc_id=pg_doc_id, total_pages=int(total_pages))
    except Exception as e:
        _log.warning("upsert_doc_progress failed: doc_id=%s err=%s", pg_doc_id, e)


def _pg_fetch_doc_state(*, ctx: IngestContext, pg_doc_id: str) -> Dict[str, Any]:
    """
    """
    out: Dict[str, Any] = {
        "has_document": False,
        "doc_title": "",
        "viewer_uri": "",
        "source_uri": "",
        "has_progress": False,
        "total_pages": 0,
        "contiguous_done_until": 0,
        "holes_total": 0,
        "holes_by_status": {},
    }

    if ctx.pg is None:
        return out
    
    try:
        with ctx.pg.cursor() as cur:
            cur.execute(
                "SELECT title, viewer_uri, source_uri FROM documents WHERE sha256=%s LIMIT 1;",
                (pg_doc_id,),
            )
            r = cur.fetchone()
            if r:
                out["has_document"] = True
                out["doc_title"] = str(r[0] or "")
                out["viewer_uri"] = str(r[1] or "")
                out["source_uri"] = str(r[2] or "")

            cur.execute(
                "SELECT total_pages, contiguous_done_until FROM doc_progress WHERE doc_id=%s LIMIT 1;",
                (pg_doc_id,),
            )
            r = cur.fetchone()
            if r:
                out["has_progress"] = True
                out["total_pages"] = int(r[0] or 0)
                out["contiguous_done_until"] = int(r[1] or 0)

            cur.execute(
                """
                SELECT status, COUNT(*)
                FROM doc_page_holes
                WHERE doc_id=%s
                GROUP BY status;
                """,
                (pg_doc_id,),
            )
            holes_by: Dict[str, int] = {}
            total = 0
            for st, cnt in cur.fetchall() or []:
                holes_by[str(st)] = int(cnt or 0)
                total += int(cnt or 0)
            out["holes_by_status"] = holes_by
            out["holes_total"] = total

    except Exception as e:
        _log.warning("PG state fetch failed. doc_id=%s err=%s", pg_doc_id, e)

    return out


def _decide_action_from_pg(*, ctx: IngestContext, pg_doc_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    """
    if ctx.pg is None:
        return "new", _pg_fetch_doc_state(ctx=ctx, pg_doc_id=pg_doc_id)
    
    state = _pg_fetch_doc_state(ctx=ctx, pg_doc_id=pg_doc_id)

    if not bool(state.get("has_progress")):
        return "new", state
    
    total_pages = int(state.get("total_pages") or 0)
    contiguous = int(state.get("contiguous_done_until") or 0)
    holes_total = int(state.get("holes_total") or 0)

    if total_pages <= 0:
        return "resume", state
    
    if contiguous >= total_pages and holes_total == 0:
        return "pass", state
    
    return "resume", state


@dataclass(frozen=True)
class ParseArtifactsResult:
    """
    """
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str

    output_dir: str
    assets_root: str
    pages_dir: str
    md_path: str

    viewer_uri: str

    text_index: str
    image_index: str
    table_index: str
    pages_staging_index: str
    images_staging_index: str
    tables_staging_index: str

    page_count: int
    staged_page_count: int
    failed_page_count: int

    extracted_image_count: int
    generated_desc_count: int
    staged_image_count: int
    indexed_image_count: int

    staged_table_count: int
    indexed_table_docs_count: int
    indexed_table_rows_count: int

    chunk_count: int
    indexed_chunk_count: int

    mode: str # new | resume | pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

def ingest_pdf(
    *,
    config_path: Optional[Path] = None,
    input_pdf: Optional[Path] = None,
    doc_id_override: Optional[str] = None,
) -> ParseArtifactsResult:
    """
    """
    ctx: Optional[IngestContext] = None
    locked = False

    try:
        ctx = build_context(config_path=config_path, source_type="pdf")
        cfg = ctx.cfg
        output_dir = ctx.output_dir

        if input_pdf is None:
            cfg_path = get_value(cfg, "paths.data_folder", None)
            cfg_pdf = get_value(cfg, "paths.input_pdf", None)
            if cfg_pdf:
                input_pdf = Path(str(cfg_path)) / cfg_pdf

        if input_pdf is None:
            raise ValueError("input_pdf is required. (set paths.input_pdf in config or pass input_pdf)")
        
        input_doc_path = Path(input_pdf)
        if not input_doc_path.exists():
            raise FileNotFoundError(f"PDF not found: {input_doc_path}")

        source_uri = str(input_doc_path)
        doc_title_local = input_doc_path.stem
        doc_sha = sha256_file(input_doc_path)

        doc_id = str(doc_id_override).strip() if doc_id_override else doc_sha
        pg_doc_id = doc_sha

        assets_root = output_dir / "assets" / doc_id
        assets_root.mkdir(parents=True, exist_ok=True)
        pages_dir = assets_root / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        pages_text_dir = assets_root / "page_texts"
        pages_text_dir.mkdir(parents=True, exist_ok=True)

        pad_ratio = float(get_value(cfg, "image_crop.pad_ratio", 0.01))
        debug_draw_bboxes = bool(get_value(cfg, "debug.draw_bboxes", False))
        render_scale = float(get_value(cfg, "render.scale", 1.0))

        filter_enabled = bool(get_value(cfg, "image_filter.enabled", False))
        stddev_min = float(get_value(cfg, "image_filter.stddev_min", 0.0))
        stddev_mode = str(get_value(cfg, "image_filter.mode", "grayscale"))
        
        md_checkpoint_every = int(get_value(cfg, "workflow.md_checkpoint_every", 1))

        action, pg_state = _decide_action_from_pg(ctx=ctx, pg_doc_id=pg_doc_id)

        if ctx.pg is not None:
            try:
                ctx.pg.reset_running_pages(doc_id=pg_doc_id, to_status=ctx.pg.HOLE_STATUS_PENDING)
            except Exception:
                pass

        if action == "pass":
            doc_title = (pg_state.get("doc_title") or doc_title_local or doc_id).strip()
            viewer_uri = str(pg_state.get("viewer_uri") or "")
            source_uri_pg = str(pg_state.get("source_uri") or "")
            if source_uri_pg:
                source_uri = source_uri_pg

            total_pages = int(pg_state.get("total_pages") or 0)
            contiguous = int(pg_state.get("contiguous_done_until") or 0)
            holes_by = pg_state.get("holes_by_status") or {}
            failed_count = int(holes_by.get("failed") or 0)

            safe_title = doc_title.strip() or doc_id
            md_cache_path = output_dir / f"{safe_title}.{doc_id[:12]}.md"

            return ParseArtifactsResult(
                doc_id=doc_id,
                doc_sha256=doc_sha,
                doc_title=doc_title,
                source_uri=source_uri,
                output_dir=str(output_dir),
                assets_root=str(assets_root),
                pages_dir=str(pages_dir),
                md_path=str(md_cache_path),
                viewer_uri=viewer_uri,
                text_index=ctx.os_text.index,
                image_index=ctx.os_image.index,
                table_index=ctx.os_table.index,
                pages_staging_index=ctx.os_pages_stage.index,
                images_staging_index=ctx.os_images_stage.index,
                tables_staging_index=ctx.os_tables_stage.index,
                page_count=total_pages,
                staged_page_count=contiguous,
                failed_page_count=failed_count,
                extracted_image_count=0,
                generated_desc_count=0,
                staged_image_count=0,
                indexed_image_count=0,
                staged_table_count=0,
                indexed_table_docs_count=0,
                indexed_table_rows_count=0,
                chunk_count=0,
                indexed_chunk_count=0,
                mode="pass"
            )
        
        viewer_uri = ""
        pdf_put: Optional[Dict[str, Any]] = None

        pdf_key = ctx.minio_writer.build_pdf_key(doc_id, filename=input_doc_path.name)
        pdf_put = ctx.minio_writer.upload_file_to_key(
            str(input_doc_path),
            object_key=pdf_key,
            content_type="application/pdf",
        )
        viewer_uri = s3_uri(pdf_put["bucket"], pdf_put["key"])

        if ctx.pg is not None:
            ctx.pg.upsert_document(
                sha256_hex=doc_sha,
                title=(doc_title_local or doc_id),
                source_uri=source_uri or None,
                viewer_uri=viewer_uri or None,
                mime_type="application/pdf",
                size_bytes=int(input_doc_path.stat().st_size),
                minio_bucket=str(pdf_put["bucket"]) if pdf_put else None,
                minio_key=str(pdf_put["key"]) if pdf_put else None,
                minio_etag=str(pdf_put["etag"]) if pdf_put else None,
            )

        page_count = 0
        staged_page_count = 0
        failed_page_count = 0

        extracted_image_count = 0
        generated_desc_count = 0
        staged_image_count = 0

        staged_table_count = 0
        indexed_table_docs_count = 0
        indexed_table_rows_count = 0

        desc_cache: Dict[str, Dict[str, str]] = {}

        if ctx.pg is not None:
            try:
                ctx.pg.advisory_lock(doc_id=pg_doc_id)
                locked = True
            except Exception as e:
                _log.warning("advisory_lock failed: doc_id=%s err=%s", pg_doc_id, e)
                raise

        try:
            _log.info(
                "Start ingest. action=%s doc_id=%s pg_doc_id=%s viewer_uri=%s",
                action,
                doc_id,
                pg_doc_id,
                viewer_uri,
            )

            page_pngs = pdf_to_page_pngs(input_doc_path, scale=render_scale)
            total_pages = len(page_pngs)
            page_count = total_pages

            _pg_init_progress_if_needed(ctx=ctx, pg_doc_id=pg_doc_id, total_pages=total_pages)

            start_page = 1
            if action == "resume" and ctx.pg is not None:
                start_page = max(1, int(ctx.pg.get_next_resume_page(doc_id=pg_doc_id)))

                _log.info("Resume info. action=%s start_page=%d total_pages=%d", action, start_page, total_pages)

            for idx, item in enumerate(tqdm(page_pngs, total=len(page_pngs), desc="OCR + IMG", unit="page"), start=1):
                page_no, payload = coerce_page_no_and_payload(item, fallback_page_no=idx)
                page_no_i = int(page_no)

                if page_no_i < start_page:
                    continue

                png_path = materialize_png_payload(payload, out_dir=pages_dir, page_no=page_no_i)
                page_id = f"{doc_id}:p{page_no_i:04d}"
                now = now_utc()

                if ctx.pg is not None:
                    ctx.pg.upsert_page_hole(
                        doc_id=pg_doc_id,
                        page_no=page_no_i,
                        status=ctx.pg.HOLE_STATUS_RUNNING,
                        attempts_inc=1,
                        last_error=None,
                    )

                try:
                    raw_txt = ocr_page(
                        png_path.read_bytes(),
                        ctx.vlm.url,
                        ctx.vlm.model,
                        ctx.vlm.api_key,
                        ctx.vlm.prompt_ocr,
                        ctx.vlm.max_tokens,
                        ctx.vlm.temperature,
                        ctx.vlm.timeout_sec,
                    )
                except Exception as e:
                    ctx.os_pages_stage.bulk_upsert(
                        [
                            {
                                "_id": page_id,
                                "_source": {
                                    "doc_id": doc_id,
                                    "page_id": page_id,
                                    "doc_title": doc_title_local,
                                    "source_uri": source_uri,
                                    "viewer_uri": viewer_uri,
                                    "source_type": "pdf",
                                    "page_no": page_no_i,
                                    "page_text": "",
                                    "ocr_model": str(ctx.vlm.model or ""),
                                    "prompt": str(ctx.vlm.prompt_ocr or ""),
                                    "status": "failed",
                                    "attempts": 1,
                                    "last_error": f"{type(e).__name__}: {e}",
                                    "created_at": now,
                                    "updated_at": now,
                                },
                            }
                        ],
                        batch_size=ctx.bulk_size,
                    )

                    if ctx.pg is not None:
                        ctx.pg.upsert_page_hole(
                            doc_id=pg_doc_id,
                            page_no=page_no_i,
                            status=ctx.pg.HOLE_STATUS_FAILED,
                            attempts_inc=0,
                            last_error=f"{type(e).__name__}: {e}",
                        )

                    _log.warning("Skip page(ocr failed): doc_id=%s page_no=%s err=%s", doc_id, page_no_i, e)
                    continue

                ref_items_raw = _parse_ref_det_items(raw_txt)
                _dump_page_items_checkpoint(pages_text_dir, page_no_i, ref_items_raw)
                table_bboxes = _collect_table_bboxes(ref_items_raw)

                try:
                    st_kwargs = dict(
                        ctx=ctx,
                        doc_id=doc_id,
                        doc_title=doc_title_local,
                        source_uri=source_uri,
                        viewer_uri=viewer_uri,
                        page_no=page_no_i,
                        page_text=raw_txt,
                        doc_sha256=doc_sha,
                    )
                    sig = inspect.signature(stage_tables_from_text)
                    if "table_bboxes" in sig.parameters:
                        st_kwargs["table_bboxes"] = table_bboxes
                    staged_table_count += stage_tables_from_text(**st_kwargs)
                except Exception as e:
                    _log.warning("Table stage failed. doc_id=%s page_no=%s err=%s", doc_id, page_no_i, e)

                new_txt_raw, img_records, desc_records, det_bbox_to_image_id = extract_and_store_images_from_page(
                    page_png_path=png_path,
                    ocr_text=raw_txt,
                    assets_root=assets_root,
                    doc_id=doc_id,
                    doc_title=doc_title_local,
                    source_uri=source_uri,
                    sha256=doc_sha,
                    page_no=page_no_i,
                    pad_ratio=pad_ratio,
                    debug_draw_bboxes=debug_draw_bboxes,
                    do_image_desc=ctx.vlm.do_image_desc,
                    img_desc_prompt=ctx.vlm.prompt_img_desc,
                    img_desc_max_tokens=ctx.vlm.img_desc_max_tokens,
                    img_desc_temperature=ctx.vlm.img_desc_temperature,
                    vlm_url=ctx.vlm.url,
                    vlm_model=ctx.vlm.model,
                    vlm_api_key=ctx.vlm.api_key,
                    vlm_timeout_sec=ctx.vlm.timeout_sec,
                    desc_cache=desc_cache,
                    filter_enabled=filter_enabled,
                    stddev_min=stddev_min,
                    stddev_mode=stddev_mode,
                    rewrite_det=False,
                )

                extracted_image_count += len(img_records or [])
                generated_desc_count += len(desc_records or [])

                final_page_text_raw = (new_txt_raw or raw_txt) or ""

                rewritten_for_clean = final_page_text_raw
                if det_bbox_to_image_id:
                    rewritten_for_clean = rewrite_image_det_bbox_to_image_id(rewritten_for_clean, det_bbox_to_image_id)
                final_page_text_clean = _strip_tags_keep_det_payload(rewritten_for_clean)

                ref_items_rewritten = _parse_ref_det_items(rewritten_for_clean)
                image_caption_bbox_by_id = _build_image_caption_bbox_map(ref_items_rewritten)

                ctx.os_pages_stage.bulk_upsert(
                    [
                        {
                            "_id": page_id,
                            "_source": {
                                "doc_id": doc_id,
                                "page_id": page_id,
                                "doc_title": doc_title_local,
                                "source_uri": source_uri,
                                "viewer_uri": viewer_uri,
                                "source_type": "pdf",
                                "page_no": page_no_i,
                                "page_text": final_page_text_clean,
                                "page_text_raw": final_page_text_raw,
                                "ocr_model": str(ctx.vlm.model or ""),
                                "prompt": str(ctx.vlm.prompt_ocr or ""),
                                "status": "done",
                                "attempts": 1,
                                "last_error": "",
                                "created_at": now,
                                "updated_at": now,
                            },
                        }
                    ],
                    batch_size=ctx.bulk_size,
                )

                if ctx.pg is not None:
                    ctx.pg.mark_page_done(doc_id=pg_doc_id, page_no=page_no_i)

                _atomic_write_text(pages_text_dir / f"page_{page_no_i:04d}.clean.txt", final_page_text_clean)
                _atomic_write_text(pages_text_dir / f"page_{page_no_i:04d}.raw.txt", final_page_text_raw)

                safe_title_initial = (doc_title_local or doc_id).strip()
                md_cache_path = output_dir / f"{safe_title_initial}.{doc_id[:12]}.md"
                if md_checkpoint_every > 0 and(page_no_i % md_checkpoint_every == 0):
                    pages_clean_local = _load_pages_clean_from_text_dir(pages_text_dir)
                    md_ckpt = _build_md_with_real_page_no(pages_clean_local)
                    _atomic_write_text(md_cache_path, md_ckpt)

                desc_by_image_id: Dict[str, Dict[str, Any]] = {
                    d.get("image_id"): d for d in (desc_records or []) if isinstance(d, dict) and d.get("image_id")
                }

                stage_docs: List[Dict[str, Any]] = []
                pg_image_rows: List[Dict[str, Any]] = []
                for order, r in enumerate(img_records or [], start=1):
                    image_id = str(r.get("image_id") or "")
                    img_path = Path(str(r.get("image_path") or ""))
                    if not image_id or not img_path.exists():
                        continue

                    stage_id = image_id
                    crop_bytes = img_path.read_bytes()
                    image_sha = sha256_bytes(crop_bytes)

                    image_key = ctx.minio_writer.build_crop_image_key(doc_id, stage_id, ext="png")
                    img_put = ctx.minio_writer.upload_bytes_to_key(
                        crop_bytes,
                        object_key=image_key,
                        content_type=r.get("image_mime") or "image/png",
                    )
                    image_uri = s3_uri(img_put["bucket"], img_put["key"])

                    desc_text = ""
                    drec = desc_by_image_id.get(image_id)
                    if drec and isinstance(drec.get("description"), str):
                        desc_text = drec["description"].strip()
                    if not desc_text:
                        desc_text = str(r.get("caption") or "").strip()
                    if not desc_text:
                        desc_text = "(no description)"

                    width, height = get_image_size_from_bytes(crop_bytes)
                    now2 = now_utc()

                    caption_bbox = image_caption_bbox_by_id.get(image_id)

                    stage_src = {
                        "doc_id": doc_id,
                        "doc_sha256": doc_sha,
                        "stage_id": stage_id,
                        "image_id": image_id,
                        "doc_title": doc_title_local,
                        "source_uri": source_uri,
                        "viewer_uri": viewer_uri,
                        "page_no": page_no_i,
                        "order": int(order),
                        "image_uri": image_uri,
                        "image_mime": r.get("image_mime") or "image/png",
                        "image_sha256": image_sha,
                        "width": int(width),
                        "height": int(height),
                        "bbox": r.get("bbox") or {},
                        "crop_bbox": r.get("crop_bbox") or {},
                        "det_bbox": r.get("det_bbox") or {},
                        "caption_bbox": caption_bbox,
                        "desc_text": desc_text,
                        "status": "pending",
                        "attempts": 0,
                        "last_error": "",
                        "created_at": now2,
                        "updated_at": now2,
                    }
                    stage_docs.append({"_id": stage_id, "_source": stage_src})
                    pg_image_rows.append(
                        {
                            "image_id": image_id,
                            "page_no": page_no_i,
                            "order": int(order),
                            "image_uri": image_uri,
                            "image_sha256": image_sha,
                            "width": int(width),
                            "height": int(height),
                            "bbox": r.get("bbox") or {},
                            "crop_bbox": r.get("crop_bbox") or {},
                            "det_bbox": r.get("det_bbox") or {},
                            "caption_bbox": caption_bbox,
                            "caption": str(r.get("caption") or ""),
                            "description": desc_text,
                        }
                    )

                if stage_docs:
                    ctx.os_images_stage.bulk_upsert(stage_docs, batch_size=ctx.bulk_size)
                    staged_image_count += len(stage_docs)
                if pg_image_rows and ctx.pg is not None:
                    ctx.pg.upsert_doc_images(doc_id=pg_doc_id, images=pg_image_rows)

            if ctx.pg is not None:
                summary = ctx.pg.get_process_summary(doc_id=pg_doc_id)
                staged_page_count = int(summary.get("done") or 0)
                failed_page_count = int(summary.get("failed") or 0)
            else:
                staged_page_count = len(list(pages_text_dir.glob("page_*.clean.txt")))
                failed_page_count = 0

        finally:
            if ctx is not None and ctx.pg is not None and locked:
                try:
                    ctx.pg.advisory_unlock(doc_id=pg_doc_id)
                except Exception:
                    pass
                locked = False

        safe_title = (doc_title_local or doc_id).strip()
        md_cache_path = output_dir / f"{safe_title}.{doc_id[:12]}.md"

        pages_clean_local = _load_pages_clean_from_text_dir(pages_text_dir)
        md_text = _build_md_with_real_page_no(pages_clean_local)
        _atomic_write_text(md_cache_path, md_text)

        chunk_count, indexed_chunk_count = index_chunks_from_md(
            ctx=ctx,
            doc_id=doc_id,
            doc_title=safe_title,
            source_uri=source_uri,
            viewer_uri=viewer_uri,
            doc_sha256=doc_sha,
            md_text=md_text,
            write_pg=False,
        )

        pages_raw_for_pg: List[Tuple[int, str]] = []
        for raw_file in sorted(pages_text_dir.glob("page_*.raw.txt")):
            m = re.search(r"page_(\d{4})\.raw\.txt$", raw_file.name)
            if not m:
                continue
            pn = int(m.group(1))
            raw = raw_file.read_text(encoding="utf-8").strip()
            if raw:
                pages_raw_for_pg.append((pn, raw))
        pages_raw_for_pg.sort(key=lambda x: x[0])

        _upsert_pg_paragraphs_sentences_from_pages(
            ctx=ctx,
            doc_sha256=doc_sha,
            pages_raw=pages_raw_for_pg,
        )

        max_rows_embed = int(get_value(cfg, "tables.max_rows_embed", 500))
        if ctx.tables_enabled:
            if staged_table_count == 0:
                try:
                    _log.info("No tables staged during OCR loop. Backfilling from pages_staging.")
                    query = {
                        "query": {"term": {"doc_id": doc_id}},
                        "sort": [{"page_no": "asc"}],
                        "_source": ["page_no", "page_text_raw"],
                    }
                    for hit in ctx.os_pages_stage.scan(query=query, size=500):
                        src = hit.get("_source") or {}
                        page_no = int(src.get("page_no") or 0)
                        raw_text = str(src.get("page_text_raw") or "")
                        if not raw_text:
                            continue
                        ref_items = _parse_ref_det_items(raw_text)
                        table_bboxes = _collect_table_bboxes(ref_items)
                        staged_table_count += stage_tables_from_text(
                            ctx=ctx,
                            doc_id=doc_id,
                            doc_title=safe_title,
                            source_uri=source_uri,
                            viewer_uri=viewer_uri,
                            page_no=page_no,
                            page_text=raw_text,
                            doc_sha256=doc_sha,
                            table_bboxes=table_bboxes,
                        )
                except Exception as e:
                    _log.warning("Table backfill failed: %s", e)

            it, ir = finalize_tables_from_staging(ctx=ctx, doc_id=doc_id, max_rows_embed=max_rows_embed)
            indexed_table_docs_count += it
            indexed_table_rows_count += ir

        emb_cfg = OllamaEmbeddingConfig(
            base_url=str(get_value(cfg, "embed_text.ollama_base_url", "http://localhost:11434")),
            model=str(get_value(cfg, "embed_text.model", "")),
            timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
            truncate=_as_bool(get_value(cfg, "embed_text.truncate", True), default=True),
        )
        if not emb_cfg.model:
            raise ValueError("embed_text.model is required. (e.g. qwen3-embedding:8b)")
        emb_provider = OllamaEmbeddingProvider(emb_cfg)
        text_embedding_model = emb_cfg.model

        text_expected_dim = int(get_value(cfg, "embed_text.expected_dim", 4096))
        if text_expected_dim != 4096:
            raise ValueError(f"text embedding dim must be 4096. got={text_expected_dim}")
        text_max_batch = int(get_value(cfg, "embed_text.max_batch_size", 32))

        img_url = get_value(cfg, "embed_image.ollama_base_url", None)
        if not img_url:
            img_url = get_value(cfg, "embed_image.base_url", "http://127.0.0.1:8088/embed")
        throttle = get_value(cfg, "embed_image.throttle_sec", None)
        if throttle is None:
            throttle = get_value(cfg, "embed_image.throtthle_sec", 0.0)

        image_embed_cfg = ImageEmbedConfig(
            url=str(img_url),
            timeout_sec=float(get_value(cfg, "embed_image.timeout_sec", 60.0)),
            expected_dim=int(get_value(cfg, "embed_image.expected_dim", 1024)),
            dimension=int(get_value(cfg, "embed_image.dimension", 1024)),
            max_images_per_request=int(get_value(cfg, "embed_image.max_images_per_request", 8)),
            retry_once=_as_bool(get_value(cfg, "embed_image.retry_once", True), default=True),
            throttle_sec=float(throttle or 0.0),
            model=str(get_value(cfg, "embed_image.model", "jinaai/jina-clip-v2")),
        )
        if image_embed_cfg.expected_dim != 1024:
            raise ValueError(f"image embedding dim must be 1024. got={image_embed_cfg.expected_dim}")
        image_embedding_model = str(get_value(cfg, "embed_image.model", "jinaai/jina-clip-v2"))

        minio_cfg = MinIOConfig(
            endpoint=str(get_value(cfg, "minio.endpoint", "")),
            access_key=str(get_value(cfg, "minio.access_key", "")),
            secret_key=str(get_value(cfg, "minio.secret_key", "")),
            bucket=str(get_value(cfg, "minio.bucket", "")),
            secure=_as_bool(get_value(cfg, "minio.secure", False), default=False),
        )
        minio_reader = MinIOReader(minio_cfg)

        _log.info("Start embedding from image staging. doc_id=%s", doc_id)

        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {"terms": {"status": ["pending", "done"]}},
                    ]
                }
            },
            "sort": [{"page_no": "asc"}, {"order": "asc"}],
        }

        indexed_image_count = 0
        batch_size = int(get_value(cfg, "embed_image.batch_size", 16))
        if batch_size <= 0:
            batch_size = 16

        buf: List[Dict[str, Any]] = []
        for hit in ctx.os_images_stage.scan(query=query, size=500):
            src = hit.get("_source", {})
            if not src:
                continue
            buf.append(src)

            if len(buf) < batch_size:
                continue

            indexed_image_count += _process_stage_batch(
                buf=buf,
                os_image=ctx.os_image,
                os_images_stage=ctx.os_images_stage,
                bulk_size=ctx.bulk_size,
                emb_provider=emb_provider,
                text_max_batch=text_max_batch,
                text_expected_dim=text_expected_dim,
                text_embedding_model=text_embedding_model,
                image_embed_cfg=image_embed_cfg,
                image_embedding_model=image_embedding_model,
                minio_reader=minio_reader,
            )
            buf = []

        if buf:
            indexed_image_count += _process_stage_batch(
                buf=buf,
                os_image=ctx.os_image,
                os_images_stage=ctx.os_images_stage,
                bulk_size=ctx.bulk_size,
                emb_provider=emb_provider,
                text_max_batch=text_max_batch,
                text_expected_dim=text_expected_dim,
                text_embedding_model=text_embedding_model,
                image_embed_cfg=image_embed_cfg,
                image_embedding_model=image_embedding_model,
                minio_reader=minio_reader,
            )

        text_index = ctx.os_text.index
        image_index = ctx.os_image.index
        table_index = ctx.os_table.index
        pages_staging_index = ctx.os_pages_stage.index
        images_staging_index = ctx.os_images_stage.index
        tables_staging_index = ctx.os_tables_stage.index

        _log.info(
            "Done. action=%s doc_id=%s pg_doc_id=%s pages(total=%d done=%d failed=%d) viewer_uri=%s "
            "images(extracted=%d staged=%d indexed=%d) chunk(total=%d indexed=%d) "
            "indexes(text=%s image=%s table=%s pages_stage=%s images_stage=%s tables_stage=%s)",
            action,
            doc_id,
            pg_doc_id,
            page_count,
            staged_page_count,
            failed_page_count,
            viewer_uri,
            extracted_image_count,
            staged_image_count,
            indexed_image_count,
            chunk_count,
            indexed_chunk_count,
            text_index,
            image_index,
            table_index,
            pages_staging_index,
            images_staging_index,
            tables_staging_index,
        )

        return ParseArtifactsResult(
            doc_id=doc_id,
            doc_sha256=doc_sha,
            doc_title=safe_title,
            source_uri=source_uri,
            output_dir=str(output_dir),
            assets_root=str(assets_root),
            pages_dir=str(pages_dir),
            md_path=str(md_cache_path),
            viewer_uri=viewer_uri,
            text_index=text_index,
            image_index=image_index,
            table_index=table_index,
            pages_staging_index=pages_staging_index,
            images_staging_index=images_staging_index,
            tables_staging_index=tables_staging_index,
            page_count=page_count,
            staged_page_count=staged_page_count,
            failed_page_count=failed_page_count,
            extracted_image_count=extracted_image_count,
            generated_desc_count=generated_desc_count,
            staged_image_count=staged_image_count,
            indexed_image_count=indexed_image_count,
            staged_table_count=staged_table_count,
            indexed_table_docs_count=indexed_table_docs_count,
            indexed_table_rows_count=indexed_table_rows_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            mode=action,
        )
    
    finally:
        if ctx is not None:
            ctx.close()


def _process_stage_batch(
    *,
    buf: List[Dict[str, Any]],
    os_image,
    os_images_stage,
    bulk_size: int,
    emb_provider: OllamaEmbeddingProvider,
    text_max_batch: int,
    text_expected_dim: int,
    text_embedding_model: str,
    image_embed_cfg: ImageEmbedConfig,
    image_embedding_model: str,
    minio_reader: MinIOReader,
) -> int:
    """images_staging의 pending 레코드를 배치로 처리해 최종 image_index에 적재합니다."""
    desc_texts = [str(b.get("desc_text") or "").strip() for b in buf]

    try:
        desc_vecs = embed_texts(
            emb_provider,
            desc_texts,
            max_batch_size=min(text_max_batch, len(desc_texts)),
            expected_dim=text_expected_dim,
        )
    except Exception as e:
        _log.warning("Desc batch embed failed. retry once. err=%s", e)
        try:
            desc_vecs = embed_texts(
                emb_provider,
                desc_texts,
                max_batch_size=min(text_max_batch, len(desc_texts)),
                expected_dim=text_expected_dim,
            )
        except Exception as e2:
            now = now_utc()
            fail_docs = []
            for b in buf:
                stage_id = b.get("stage_id") or b.get("image_id")
                fail_docs.append({
                    "_id": stage_id,
                    "_source": {
                        **b,
                        "status": "failed",
                        "attempts": int(b.get("attempts", 0)) + 1,
                        "last_error": f"desc_embed_failed: {e2}",
                        "updated_at": now,
                    },
                })
            os_images_stage.bulk_upsert(fail_docs, batch_size=bulk_size)
            return 0

    if len(desc_vecs) != len(buf):
        now = now_utc()
        fail_docs = []
        for b in buf:
            stage_id = b.get("stage_id") or b.get("image_id")
            fail_docs.append({
                "_id": stage_id,
                "_source": {
                    **b,
                    "status": "failed",
                    "attempts": int(b.get("attempts", 0)) + 1,
                    "last_error": f"desc_embed_len_mismatch: got={len(desc_vecs)} expected={len(buf)}",
                    "updated_at": now,
                },
            })
        os_images_stage.bulk_upsert(fail_docs, batch_size=bulk_size)
        return 0

    try:
        image_bytes_list: List[bytes] = _fetch_images_from_minio(buf=buf, minio_reader=minio_reader)
        if len(image_bytes_list) != len(buf):
            raise RuntimeError(f"image_fetch_len_mismatch: got={len(image_bytes_list)} expected={len(buf)}")

        img_vecs = embed_images_bytes_batch(image_bytes_list, cfg=image_embed_cfg)

        if len(img_vecs) != len(buf):
            raise RuntimeError(f"image_embed_len_mismatch: got={len(img_vecs)} expected={len(buf)}")

    except Exception as e:
        now = now_utc()
        fail_docs = []
        for b in buf:
            stage_id = b.get("stage_id") or b.get("image_id")
            fail_docs.append({
                "_id": stage_id,
                "_source": {
                    **b,
                    "status": "failed",
                    "attempts": int(b.get("attempts", 0)) + 1,
                    "last_error": f"image_embed_failed: {e}",
                    "updated_at": now,
                },
            })
        os_images_stage.bulk_upsert(fail_docs, batch_size=bulk_size)
        return 0

    now = now_utc()
    image_docs: List[Dict[str, Any]] = []
    done_docs: List[Dict[str, Any]] = []

    for b, dvec, ivec in zip(buf, desc_vecs, img_vecs):
        image_id = b["image_id"]
        stage_id = b.get("stage_id") or image_id

        src = {
            "doc_id": b["doc_id"],
            "image_id": image_id,
            "stage_id": stage_id,
            "doc_title": b.get("doc_title") or "",
            "source_uri": b.get("source_uri") or "",
            "page_no": int(b.get("page_no") or 0),
            "order": int(b.get("order") or 0),
            "pdf_uri": b.get("pdf_uri") or "",
            "image_uri": b.get("image_uri") or "",
            "image_mime": b.get("image_mime") or "image/png",
            "image_sha256": b.get("image_sha256") or "",
            "width": int(b.get("width") or 0),
            "height": int(b.get("height") or 0),
            "bbox": b.get("bbox") or {},
            "desc_text": b.get("desc_text") or "",
            "desc_embedding": dvec,
            "image_embedding": ivec,
            "desc_embedding_model": text_embedding_model,
            "image_embedding_model": image_embedding_model,
            "ingested_at": now,
        }
        image_docs.append({"_id": stage_id, "_source": src})

        done_docs.append({
            "_id": stage_id,
            "_source": {
                **b,
                "status": "done",
                "attempts": int(b.get("attempts", 0)) + 1,
                "last_error": "",
                "updated_at": now,
            }
        })

    os_image.bulk_upsert(image_docs, batch_size=bulk_size)
    os_images_stage.bulk_upsert(done_docs, batch_size=bulk_size)

    return len(image_docs)


def _fetch_images_from_minio(*, buf: List[Dict[str, Any]], minio_reader: MinIOReader) -> List[bytes]:
    """staging 레코드들의 image_uri를 MinIO에서 다운로드해 bytes 리스트로 반환합니다."""
    images: List[bytes] = []
    for b in buf:
        uri = str(b.get("image_uri") or "")
        bucket, key = parse_s3_uri(uri)
        images.append(minio_reader.download_bytes(bucket=bucket, key=key))
    return images
