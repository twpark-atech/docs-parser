import hashlib
import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from app.common.hash import sha256_file, sha256_bytes
from app.common.parser import get_value, pdf_to_page_pngs
from app.common.runtime import now_utc
from app.pipelines.document_ingest import build_context
from app.parsing.pdf import coerce_page_no_and_payload, materialize_png_payload
from app.parsing.ocr import ocr_page
from app.parsing.regex import RE_CAP_BOX_NAMED, RE_IMAGE_BOX_NAMED, RE_TABLE_BLOCK_WITH_HTML
from app.parsing.table_extractor import build_table_id, normalize_table_html

_log = logging.getLogger(__name__)


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def _source_name_for_object_key(source_name: str, default_ext: str) -> tuple[str, str]:
    raw = (source_name or "").strip()
    # Handle both POSIX/Windows separators from incoming filename strings.
    base = re.split(r"[\\/]", raw)[-1].strip() if raw else ""
    if not base:
        base = f"document{default_ext}"
    stem = Path(base).stem.strip() or "document"
    ext = Path(base).suffix.lstrip(".").strip() or default_ext.lstrip(".")
    return f"{stem}_{ext}", base


def _to_bbox_float(m: re.Match) -> tuple[float, float, float, float]:
    x1 = float(m.group("x1"))
    y1 = float(m.group("y1"))
    x2 = float(m.group("x2"))
    y2 = float(m.group("y2"))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _image_id(doc_id: str, page_no: int, bbox: tuple[int, int, int, int]) -> str:
    key = f"{doc_id}|p{page_no:04d}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{page_no:04d}:img{h}"


def _infer_bbox_scale(boxes: list[tuple[float, float, float, float]], w: int, h: int) -> tuple[float, float]:
    if not boxes:
        return 1.0, 1.0

    max_x2 = max(b[2] for b in boxes)
    max_y2 = max(b[3] for b in boxes)

    if max_x2 <= 1.5 and max_y2 <= 1.5:
        return float(w), float(h)

    if max_x2 <= 1100 and max_y2 <= 1100 and (w > 1200 or h > 1200):
        return w / 1000.0, h / 1000.0

    sx = 1.0
    sy = 1.0
    if max_x2 > 0 and (max_x2 < w * 0.85 or max_x2 > w * 1.15):
        sx = w / float(max_x2)
    if max_y2 > 0 and (max_y2 < h * 0.85 or max_y2 > h * 1.15):
        sy = h / float(max_y2)
    return sx, sy


def _transform_bbox(
    bbox: tuple[float, float, float, float],
    *,
    sx: float,
    sy: float,
    w: int,
    h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    fx1 = int(round(x1 * sx))
    fy1 = int(round(y1 * sy))
    fx2 = int(round(x2 * sx))
    fy2 = int(round(y2 * sy))

    bw = max(1, fx2 - fx1)
    bh = max(1, fy2 - fy1)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))

    fx1 = max(0, min(fx1 - px, w))
    fy1 = max(0, min(fy1 - py, h))
    fx2 = max(0, min(fx2 + px, w))
    fy2 = max(0, min(fy2 + py, h))
    return fx1, fy1, fx2, fy2


def _replace_and_store_entities(
    *,
    text: str,
    doc_id: str,
    page_no: int,
    png_path: Path,
    pad_ratio: float,
    minio_writer,
    minio_prefix: str,
) -> str:
    if not text:
        return text

    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    cap_candidates: list[tuple[tuple[float, float, float, float], str]] = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        mcap = RE_CAP_BOX_NAMED.search(ln)
        if not mcap:
            continue
        try:
            bx1, by1, bx2, by2 = (
                float(mcap.group("x1")),
                float(mcap.group("y1")),
                float(mcap.group("x2")),
                float(mcap.group("y2")),
            )
        except Exception:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        cap_text = lines[j].strip() if j < len(lines) else ""
        cap_candidates.append(((bx1, by1, bx2, by2), cap_text))

    table_order = 0

    def _table_repl(m: re.Match) -> str:
        nonlocal table_order
        table_order += 1
        raw_html = normalize_table_html(m.group("html"))
        table_id, _ = build_table_id(doc_id=doc_id, page_no=page_no, order=table_order, raw_html=raw_html)
        safe_file = f"{_safe_name(table_id)}.md"
        table_payload = (raw_html + "\n").encode("utf-8")
        if minio_writer is not None and minio_prefix:
            try:
                minio_writer.upload_bytes_to_key(
                    table_payload,
                    object_key=f"{minio_prefix}/tables/{safe_file}",
                    content_type="text/markdown; charset=utf-8",
                )
            except Exception as e:
                _log.warning("MinIO table upload failed: %s", e)
        return f"[TABLE_ID:{table_id}]"

    out = RE_TABLE_BLOCK_WITH_HTML.sub(_table_repl, text)

    image_matches = list(RE_IMAGE_BOX_NAMED.finditer(out))
    raw_boxes = [_to_bbox_float(m) for m in image_matches]
    sx, sy = _infer_bbox_scale(raw_boxes, w, h)
    raw_box_map = {
        (m.start(), m.end()): _transform_bbox(
            _to_bbox_float(m),
            sx=sx,
            sy=sy,
            w=w,
            h=h,
            pad_ratio=0.0,
        )
        for m in image_matches
    }
    crop_box_map = {
        (m.start(), m.end()): _transform_bbox(
            _to_bbox_float(m),
            sx=sx,
            sy=sy,
            w=w,
            h=h,
            pad_ratio=pad_ratio,
        )
        for m in image_matches
    }

    def _image_repl(m: re.Match) -> str:
        x1f, y1f, x2f, y2f = _to_bbox_float(m)
        if abs(x1f) < 1e-6 and abs(y1f) < 1e-6 and x2f >= 0.99 and y2f >= 0.99:
            return ""
        if abs(x1f) < 1e-6 and abs(y1f) < 1e-6 and x2f >= 999 and y2f >= 999:
            return ""

        raw_bbox = raw_box_map.get((m.start(), m.end()))
        crop_bbox = crop_box_map.get((m.start(), m.end()))
        if raw_bbox is None or crop_bbox is None:
            return ""

        image_id = _image_id(doc_id, page_no, raw_bbox)
        cap_text_best = ""
        if cap_candidates:
            ix = (x1f + x2f) / 2.0
            iy = (y1f + y2f) / 2.0
            best_d = None
            for (bx1, by1, bx2, by2), cap_text in cap_candidates:
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                d = (ix - cx) ** 2 + (iy - cy) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    cap_text_best = cap_text

        x1, y1, x2, y2 = crop_bbox
        if (x2 - x1) >= 8 and (y2 - y1) >= 8:
            crop = img.crop((x1, y1, x2, y2))
            safe_file = f"{_safe_name(image_id)}.png"
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            image_key = f"{minio_prefix}/images/{safe_file}"
            try:
                minio_writer.upload_bytes_to_key(
                    png_bytes,
                    object_key=image_key,
                    content_type="image/png",
                )
            except Exception as e:
                _log.warning("MinIO image upload failed: %s", e)
            return f"[IMAGE_ID:{image_id}]"
        return ""

    out = RE_IMAGE_BOX_NAMED.sub(_image_repl, out)
    return out


def _stage_page(
    *,
    ctx,
    doc_id: str,
    page_no: int,
    doc_title: str,
    source_uri: str,
    pdf_uri: str,
    ocr_text: str,
    ocr_model: str,
    prompt: str,
    status: str,
    attempts: int,
    last_error: str = "",
) -> None:
    page_id = f"{doc_id}:p{int(page_no):04d}"
    now = now_utc()
    prompt_sha = sha256_bytes(prompt.encode("utf-8", errors="ignore")) if prompt else ""
    doc = {
        "_id": page_id,
        "_source": {
            "doc_id": doc_id,
            "page_id": page_id,
            "doc_title": doc_title,
            "source_uri": source_uri,
            "pdf_uri": pdf_uri,
            "page_no": int(page_no),
            "ocr_text": ocr_text,
            "ocr_model": ocr_model,
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
            "status": status,
            "attempts": int(attempts),
            "last_error": last_error,
            "created_at": now,
            "updated_at": now,
        },
    }
    ctx.os_pages_stage.bulk_upsert([doc], batch_size=1)


def _load_staged_pages(*, ctx, doc_id: str) -> list[str]:
    query = {"query": {"term": {"doc_id": str(doc_id)}}}
    rows = []
    for hit in ctx.os_pages_stage.scan(query=query, size=500):
        src = hit.get("_source", {}) if isinstance(hit, dict) else {}
        if str(src.get("status") or "").lower() != "done":
            continue
        rows.append((int(src.get("page_no") or 0), str(src.get("ocr_text") or "")))
    rows.sort(key=lambda x: x[0])
    return [txt for _, txt in rows]


class PDFParser:
    @staticmethod
    def parse(
        *,
        config_path: Optional[Path] = None,
        input_pdf: Optional[Path] = None,
        doc_id_override: Optional[str] = None,
        source_name: Optional[str] = None,
    ):
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

        doc_sha = sha256_file(input_doc_path)
        doc_id = str(doc_id_override).strip() if doc_id_override else doc_sha
        source_filename = source_name or input_doc_path.name
        minio_prefix, minio_original_name = _source_name_for_object_key(source_filename, ".pdf")

        render_scale = float(get_value(cfg, "render.scale", 1.0))
        pad_ratio = float(get_value(cfg, "image_crop.pad_ratio", 0.01))
        page_pngs = pdf_to_page_pngs(input_doc_path, scale=render_scale)
        original_key = f"{minio_prefix}/{minio_original_name}"
        original_upload: dict = {}
        try:
            original_upload = ctx.minio_writer.upload_file_to_key(
                str(input_doc_path),
                object_key=original_key,
                content_type="application/pdf",
            )
        except Exception as e:
            _log.warning("MinIO original upload failed: %s", e)

        resume_page = 1
        if ctx.pg is not None:
            try:
                ctx.pg.upsert_document(
                    sha256_hex=doc_id,
                    title=Path(source_filename).stem,
                    source_uri=source_filename,
                    viewer_uri=f"minio://{str(get_value(cfg, 'minio.bucket', ''))}/{original_key}",
                    mime_type="application/pdf",
                    size_bytes=int(input_doc_path.stat().st_size),
                    minio_bucket=str(get_value(cfg, "minio.bucket", "")),
                    minio_key=original_key,
                    minio_etag=str(original_upload.get("etag") or ""),
                )
                ctx.pg.upsert_doc_progress(doc_id=doc_id, total_pages=len(page_pngs))
                ctx.pg.reset_running_pages(doc_id=doc_id, to_status=ctx.pg.HOLE_STATUS_PENDING)
                resume_page = int(ctx.pg.get_next_resume_page(doc_id=doc_id))
            except Exception as e:
                _log.warning("PG progress init skipped/failed: %s", e)

        # PG checkpoint와 별개로 pages_staging의 미완료 페이지가 남아 있으면
        # 해당 최소 페이지부터 재시도합니다.
        try:
            q = {"query": {"term": {"doc_id": str(doc_id)}}}
            min_not_done: Optional[int] = None
            for hit in ctx.os_pages_stage.scan(query=q, size=500):
                src = hit.get("_source", {}) if isinstance(hit, dict) else {}
                st = str(src.get("status") or "").lower()
                if st == "done":
                    continue
                p = int(src.get("page_no") or 0)
                if p <= 0:
                    continue
                if min_not_done is None or p < min_not_done:
                    min_not_done = p
                if min_not_done is not None:
                    resume_page = min(int(resume_page), int(min_not_done))
        except Exception as e:
            _log.warning("pages_staging resume scan skipped/failed: doc_id=%s err=%s", doc_id, e)

        with tempfile.TemporaryDirectory(prefix="docs-parser-pages-") as tmpdir:
            pages_dir = Path(tmpdir)
            for idx, item in enumerate(tqdm(page_pngs, total=len(page_pngs), desc="OCR + IMG", unit="page"), start=1):
                page_no, payload = coerce_page_no_and_payload(item, fallback_page_no=idx)
                page_no_i = int(page_no)
                if page_no_i < resume_page:
                    continue
                raw_txt = ""
                processed = ""
                last_error = ""
                status = "done"
                try:
                    png_path = materialize_png_payload(payload, out_dir=pages_dir, page_no=page_no_i)
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
                    processed = _replace_and_store_entities(
                        text=raw_txt or "",
                        doc_id=doc_id,
                        page_no=page_no_i,
                        png_path=png_path,
                        pad_ratio=pad_ratio,
                        minio_writer=ctx.minio_writer,
                        minio_prefix=minio_prefix,
                    )
                except Exception as e:
                    status = "failed"
                    last_error = str(e)
                    _log.warning("Page process error: page_no=%s err=%s", page_no_i, e)

                try:
                    _stage_page(
                        ctx=ctx,
                        doc_id=doc_id,
                        page_no=page_no_i,
                        doc_title=Path(source_filename).stem,
                        source_uri=source_filename,
                        pdf_uri=f"minio://{str(get_value(cfg, 'minio.bucket', ''))}/{original_key}",
                        ocr_text=processed or "",
                        ocr_model=ctx.vlm.model,
                        prompt=ctx.vlm.prompt_ocr,
                        status=status,
                        attempts=1,
                        last_error=last_error,
                    )
                except Exception as e:
                    _log.warning("pages_staging upsert failed: page_no=%s err=%s", page_no_i, e)

                if ctx.pg is not None:
                    try:
                        if status == "done":
                            ctx.pg.mark_page_done(doc_id=doc_id, page_no=page_no_i)
                        else:
                            ctx.pg.upsert_page_hole(
                                doc_id=doc_id,
                                page_no=page_no_i,
                                status=ctx.pg.HOLE_STATUS_FAILED,
                                attempts_inc=1,
                                last_error=last_error,
                            )
                    except Exception as e:
                        _log.warning("PG checkpoint update skipped/failed: page_no=%s err=%s", page_no_i, e)

        staged_pages = _load_staged_pages(ctx=ctx, doc_id=doc_id)
        title = Path(source_filename).stem
        minio_bucket = str(get_value(cfg, "minio.bucket", ""))
        return {
            "doc_id": doc_id,
            "title": title,
            "source_name": source_filename,
            "source_uri": source_filename,
            "viewer_uri": f"minio://{minio_bucket}/{original_key}" if minio_bucket else "",
            "mime_type": "application/pdf",
            "size_bytes": int(input_doc_path.stat().st_size),
            "minio_bucket": minio_bucket,
            "minio_key": original_key,
            "minio_etag": str(original_upload.get("etag") or ""),
            "pages": staged_pages,
        }
