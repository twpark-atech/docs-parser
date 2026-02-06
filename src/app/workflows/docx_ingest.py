# ==============================================================================
# 목적 : DOCX 파싱 + OpenSearch/Postgres/MinIO 적재
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.common.config import load_config
from app.common.hash import sha256_file, sha256_bytes
from app.common.ids import s3_uri
from app.common.parser import get_value
from app.common.runtime import now_utc
from app.common.converter import (
    normalize_table_html_to_tr_td,
    ProvenanceEntry,
    DocxMediaImage,
    ExportedImage,
    make_docx_image_id,
)

from app.parsing.regex import RE_BULLET, RE_NUM, RE_HTML_TABLE, RE_KIND_BBOX, RE_HTML_TAG, RE_PIPE_TAG
from app.parsing.table_extractor import extract_html_tables, parse_table
from app.parsing.image_extractor import describe_image_short, get_image_size_from_bytes
from app.storage.embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider
from app.indexing.embedding import ImageEmbedConfig
from app.storage.minio import MinIOReader, MinIOConfig
from app.workflows import pdf_ingest as pdf_ingest_mod
from app.workflows.common_ingest import (
    build_context,
    stage_tables_from_text,
    finalize_tables_from_staging,
    index_chunks_from_md,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseDocxResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str
    output_dir: str
    md_path: str
    provenance_path: str
    images_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocxIngestResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str
    output_dir: str
    md_path: str
    provenance_path: str
    images_dir: str
    viewer_uri: str
    text_index: str
    table_index: str
    tables_staging_index: str
    images_staging_index: str
    chunk_count: int
    indexed_chunk_count: int
    staged_table_count: int
    indexed_table_docs_count: int
    indexed_table_rows_count: int
    staged_image_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def restore_lists(md: str) -> str:
    out: List[str] = []
    for line in md.splitlines():
        m = RE_BULLET.match(line)
        if m:
            indent, _, body = m.groups()
            level = max(0, len(indent) // 2)
            out.append(" " * level + f"- {body}")
            continue

        m = RE_NUM.match(line)
        if m:
            indent, num, body = m.groups()
            level = max(0, len(indent) // 2)
            out.append(" " * level + f"{num}. {body}")
            continue

        out.append(line)
    return "\n".join(out)


def normalize_tables_in_md(md_text: str) -> str:
    if not md_text or "<table" not in md_text.lower():
        return md_text

    def repl(m: re.Match) -> str:
        raw = m.group(1)
        out = normalize_table_html_to_tr_td(raw, keep_br=True)
        return out.replace("\r", "").replace("\n", "")

    return RE_HTML_TABLE.sub(repl, md_text)


def _clean_md_for_chunks(md_text: str) -> str:
    if not md_text:
        return ""
    s = RE_KIND_BBOX.sub("", md_text)
    s = RE_PIPE_TAG.sub("", s)
    s = RE_HTML_TAG.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _ref_line(ref: str, det: Optional[str] = None) -> str:
    # DOCX has no bbox; emit dummy bbox to match text[[x1,y1,x2,y2]] style.
    if det and re.match(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$", det):
        return f"{ref}[[{det}]]"
    return f"{ref}[[0, 0, 0, 0]]"


def extract_docx_media(docx_path: Path) -> List[DocxMediaImage]:
    out: List[DocxMediaImage] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        names = [n for n in zf.namelist() if n.startswith("word/media/")]
        for name in sorted(names):
            try:
                b = zf.read(name)
            except Exception:
                continue

            ext = Path(name).suffix.lower().lstrip(".") or "bin"
            out.append(DocxMediaImage(name=name, bytes=b, ext=ext))
    return out


def _sha1_bytes16(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()[:16]


def _save_docx_media_image_as_png(image_bytes: bytes, out_png: Path) -> None:
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes))
    img.save(out_png, format="PNG")


def save_all_docx_media_images(
    *,
    doc_id: str,
    docx_media: List[DocxMediaImage],
    images_dir: Path,
    images_subdir_name: str = "images",
) -> List[ExportedImage]:
    images_dir.mkdir(parents=True, exist_ok=True)

    exported: List[ExportedImage] = []
    for idx, m in enumerate(docx_media, start=1):
        key = f"{idx}:{m.name}:{_sha1_bytes16(m.bytes)}"
        image_id = make_docx_image_id(doc_id=doc_id, page_no=None, key=key)

        safe_name = image_id.replace(":", "_")
        out_png = images_dir / f"{safe_name}.png"
        out_bin = images_dir / f"{safe_name}.bin"

        if not out_png.exists() and not out_bin.exists():
            try:
                _save_docx_media_image_as_png(m.bytes, out_png)
            except Exception:
                out_bin.write_bytes(m.bytes)

        if out_png.exists():
            rel_path = f"{images_subdir_name}/{out_png.name}"
        else:
            rel_path = f"{images_subdir_name}/{out_bin.name}"

        exported.append(
            ExportedImage(
                image_id=image_id,
                rel_path=rel_path,
                page_no=None,
                bbox=None,
                location_tokens=None,
            )
        )
    return exported


_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _open_docx_document_xml(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        return zf.read("word/document.xml").decode("utf-8", errors="replace")


def _get_para_text(p: ET.Element) -> str:
    parts: List[str] = []
    for node in p.iter():
        tag = node.tag
        if tag.endswith("}t") and node.text:
            parts.append(node.text)
        elif tag.endswith("}tab"):
            parts.append("\t")
        elif tag.endswith("}br"):
            parts.append("\n")
    s = "".join(parts)
    return s.strip()


def _cell_text(tc: ET.Element) -> str:
    paras = tc.findall(".//w:p", _W_NS)
    lines = []
    for p in paras:
        t = _get_para_text(p)
        if t:
            lines.append(t)
    return "\n".join(lines).strip()


def _table_to_html_from_xml(tbl: ET.Element) -> str:
    rows: List[List[str]] = []

    for tr in tbl.findall(".//w:tr", _W_NS):
        row: List[str] = []
        tcs = tr.findall("./w:tc", _W_NS)
        for tc in tcs:
            txt = _cell_text(tc)
            if txt:
                txt = "<br/>".join([x.strip() for x in txt.split("\n") if x.strip()])
            row.append(txt)
        if row:
            rows.append(row)

    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 0:
        return "<table></table>"

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: List[str] = ["<table>"]
    for r in rows:
        tds = [f"<td>{c}</td>" for c in r]
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</table>")

    return normalize_table_html_to_tr_td("".join(parts), keep_br=True).replace("\r", "").replace("\n", "")


def export_grounded_md_from_docx_xml(
    *,
    docx_path: Path,
    doc_id: str,
    provenance_out: Optional[List[ProvenanceEntry]] = None,
) -> str:
    xml = _open_docx_document_xml(docx_path)
    root = ET.fromstring(xml)

    body = root.find(".//w:body", _W_NS)
    if body is None:
        return ""

    lines: List[str] = []
    table_idx = 0

    for child in list(body):
        tag = child.tag
        if tag.endswith("}p"):
            txt = _get_para_text(child)
            if not txt:
                continue

            lines.append(_ref_line("text"))
            lines.append(txt)
            lines.append("")

            if provenance_out is not None:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="text",
                        page_no=None,
                        bbox=None,
                        loc_tokens=None,
                        ref="text",
                        text=txt,
                    )
                )

        elif tag.endswith("}tbl"):
            table_idx += 1
            html = _table_to_html_from_xml(child)
            table_id = f"docx_table_{table_idx:04d}"

            lines.append(_ref_line("table", table_id))
            lines.append(html)
            lines.append("")

            if provenance_out is not None:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="table",
                        page_no=None,
                        bbox=None,
                        loc_tokens=None,
                        ref="table",
                        text=None,
                    )
                )
        else:
            # 기타 요소는 일단 무시
            continue

    return "\n".join(lines).strip()


def ingest_docx(
    *,
    config_path: Optional[Path] = None,
    storage_config_path: Optional[Path] = None,
    input_docx: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> DocxIngestResult:
    cfg = {}
    if config_path is not None:
        cfg = load_config(config_path)
    elif output_dir is None:
        cfg = load_config(Path("config/docx_config.yml"))

    input_doc_path: Optional[Path] = input_docx
    if input_doc_path is None:
        data_folder = Path(get_value(cfg, "paths.data_folder", "."))
        input_docx_name = str(get_value(cfg, "paths.input_docx", "")).strip()
        if input_docx_name:
            input_doc_path = data_folder / input_docx_name

    if input_doc_path is None:
        raise ValueError("input_docx is required. (arg input_docx or paths.input_docx in docx_config.yml)")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"DOCX not found: {input_doc_path}")
    if input_doc_path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx is supported. got={input_doc_path.suffix}")

    if output_dir is None:
        output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_uri = str(input_doc_path)
    doc_title = input_doc_path.stem
    doc_sha = sha256_file(input_doc_path)
    doc_id = doc_sha

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md_cache_path = output_dir / f"{doc_title}.{doc_id[:12]}.md"
    prov_path = output_dir / f"{doc_title}.{doc_id[:12]}.provenance.json"

    _log.info("Start DOCX -> MD(XML-ordered). doc_id=%s source=%s", doc_id, source_uri)

    provenance: List[ProvenanceEntry] = []

    md_text = export_grounded_md_from_docx_xml(
        docx_path=input_doc_path,
        doc_id=doc_id,
        provenance_out=provenance,
    )
    _log.info("XML-ordered grounded MD built. chars=%d", len(md_text))

    md_text = restore_lists(md_text)

    md_text = normalize_tables_in_md(md_text)

    docx_media = extract_docx_media(input_doc_path)
    _log.info("Extracted DOCX media images: %d", len(docx_media))
    exported_images = save_all_docx_media_images(
        doc_id=doc_id,
        docx_media=docx_media,
        images_dir=images_dir,
        images_subdir_name="images",
    )

    md_cache_path.write_text(md_text, encoding="utf-8")

    prov_payload = {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "source_uri": source_uri,
        "exported_images": [asdict(e) for e in exported_images],
        "items": [p.to_dict() for p in provenance],
    }
    prov_path.write_text(json.dumps(prov_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _log.info(
        "Done DOCX -> MD. md_path=%s prov_path=%s images_dir=%s prov_items=%d exported_images=%d",
        str(md_cache_path),
        str(prov_path),
        str(images_dir),
        len(provenance),
        len(exported_images),
    )

    storage_cfg = storage_config_path or Path("config/config.yml")
    ctx = build_context(config_path=storage_cfg, source_type="docx")

    if exported_images and ctx.vlm.do_image_desc:
        desc_lines: List[str] = []
        for e in exported_images:
            img_path = images_dir / Path(e.rel_path).name
            if not img_path.exists():
                continue
            try:
                desc = describe_image_short(
                    img_path.read_bytes(),
                    vlm_url=ctx.vlm.url,
                    vlm_model=ctx.vlm.model,
                    vlm_api_key=ctx.vlm.api_key or "",
                    timeout_sec=ctx.vlm.timeout_sec,
                    prompt=ctx.vlm.prompt_img_desc,
                    max_tokens=ctx.vlm.img_desc_max_tokens,
                    temperature=ctx.vlm.img_desc_temperature,
                )
            except Exception:
                desc = ""
            desc_lines.append("image[[0, 0, 0, 0]]")
            if desc:
                desc_lines.append(desc)
            desc_lines.append("")
        if desc_lines:
            md_text = (md_text.rstrip() + "\n\n" + "\n".join(desc_lines)).strip() + "\n"

    viewer_uri = ""
    if ctx is not None:
        obj_key = ctx.minio_writer.build_object_key(doc_sha, filename=input_doc_path.name)
        put = ctx.minio_writer.upload_file_to_key(
            str(input_doc_path),
            object_key=obj_key,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        viewer_uri = s3_uri(put["bucket"], put["key"])

        if ctx.pg is not None:
            ctx.pg.upsert_document(
                sha256_hex=doc_sha,
                title=(doc_title or doc_id),
                source_uri=source_uri or None,
                viewer_uri=viewer_uri or None,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size_bytes=int(input_doc_path.stat().st_size),
                minio_bucket=str(put["bucket"]) if put else None,
                minio_key=str(put["key"]) if put else None,
                minio_etag=str(put["etag"]) if put else None,
            )

    staged_table_count = stage_tables_from_text(
        ctx=ctx,
        doc_id=doc_id,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        viewer_uri=viewer_uri,
        page_no=1,
        page_text=md_text,
        doc_sha256=doc_sha,
    )
    if staged_table_count == 0 and "<table" in md_text.lower():
        try:
            htmls = extract_html_tables(md_text)
            now = now_utc()
            os_docs: List[Dict[str, Any]] = []
            for order, raw_html in enumerate(htmls, start=1):
                header, rows = parse_table(raw_html)
                if not header:
                    continue
                table_id = f"{doc_id}:t0001:{order:04d}"
                table_sha = sha256_bytes(raw_html.encode("utf-8", errors="ignore"))
                if ctx.pg is not None:
                    with ctx.pg.cursor() as cur:
                        ctx.pg.upsert_pg_table(
                            cur,
                            table_id=table_id,
                            doc_id=doc_sha,
                            page_no=1,
                            order=order,
                            bbox=None,
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
                            "doc_sha256": doc_sha,
                            "table_id": table_id,
                            "doc_title": doc_title or doc_id,
                            "source_uri": source_uri,
                            "viewer_uri": viewer_uri,
                            "page_no": 1,
                            "order": order,
                            "bbox": None,
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
            if os_docs:
                ctx.os_tables_stage.bulk_upsert(os_docs, batch_size=ctx.bulk_size)
                staged_table_count = len(os_docs)
        except Exception as e:
            _log.warning("DOCX table fallback staging failed: %s", e)

    indexed_table_docs_count = 0
    indexed_table_rows_count = 0
    if ctx.tables_enabled:
        it, ir = finalize_tables_from_staging(ctx=ctx, doc_id=doc_id)
        indexed_table_docs_count += it
        indexed_table_rows_count += ir

    cleaned_md = _clean_md_for_chunks(md_text)
    chunk_count, indexed_chunk_count = index_chunks_from_md(
        ctx=ctx,
        doc_id=doc_id,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        viewer_uri=viewer_uri,
        doc_sha256=doc_sha,
        md_text=cleaned_md,
        write_pg=False,
    )

    staged_image_count = 0
    if exported_images:
        now = now_utc()
        os_docs = []
        pg_images = []
        for e in exported_images:
            img_path = images_dir / Path(e.rel_path).name
            if not img_path.exists():
                continue
            img_bytes = img_path.read_bytes()
            img_sha = sha256_bytes(img_bytes)
            w, h = get_image_size_from_bytes(img_bytes)
            obj_key = ctx.minio_writer.build_crop_image_key(doc_id, e.image_id, ext="png")
            put = ctx.minio_writer.upload_bytes_to_key(
                img_bytes,
                object_key=obj_key,
                content_type="image/png",
            )
            desc_text = ""
            if ctx.vlm.url and ctx.vlm.model:
                try:
                    desc_text = describe_image_short(
                        img_bytes,
                        vlm_url=ctx.vlm.url,
                        vlm_model=ctx.vlm.model,
                        vlm_api_key=ctx.vlm.api_key or "",
                        timeout_sec=ctx.vlm.timeout_sec,
                        prompt=ctx.vlm.prompt_img_desc,
                        max_tokens=ctx.vlm.img_desc_max_tokens,
                        temperature=ctx.vlm.img_desc_temperature,
                    )
                except Exception:
                    desc_text = ""
            os_docs.append(
                {
                    "_id": f"{doc_id}:{e.image_id}",
                    "_source": {
                        "doc_id": doc_id,
                        "stage_id": f"{doc_id}:{e.image_id}",
                        "image_id": e.image_id,
                        "doc_title": doc_title or doc_id,
                        "source_uri": source_uri,
                        "viewer_uri": viewer_uri,
                        "source_type": "docx",
                        "page_no": e.page_no or 0,
                        "order": 0,
                        "image_uri": s3_uri(put["bucket"], put["key"]),
                        "image_mime": "image/png",
                        "image_sha256": img_sha,
                        "width": int(w),
                        "height": int(h),
                        "bbox": None,
                        "desc_text": desc_text,
                        "status": "pending",
                        "attempts": 0,
                        "last_error": "",
                        "created_at": now,
                        "updated_at": now,
                    },
                }
            )
            pg_images.append(
                {
                    "image_id": e.image_id,
                    "page_no": e.page_no or 0,
                    "order": 0,
                    "image_uri": s3_uri(put["bucket"], put["key"]),
                    "image_sha256": img_sha,
                    "width": int(w),
                    "height": int(h),
                    "bbox": None,
                    "crop_bbox": None,
                    "det_bbox": None,
                    "caption_bbox": None,
                    "caption": "",
                    "description": desc_text,
                }
            )
        if os_docs:
            ctx.os_images_stage.bulk_upsert(os_docs, batch_size=ctx.bulk_size)
            staged_image_count = len(os_docs)
        if ctx.pg is not None and pg_images:
            ctx.pg.upsert_doc_images(doc_id=doc_sha, images=pg_images)

        # Finalize images: embeddings + image index
        if staged_image_count > 0:
            emb_cfg = OllamaEmbeddingConfig(
                base_url=str(get_value(ctx.cfg, "embed_text.ollama_base_url", "http://localhost:11434")),
                model=str(get_value(ctx.cfg, "embed_text.model", "")),
                timeout_sec=int(get_value(ctx.cfg, "embed_text.timeout_sec", 120)),
                truncate=bool(get_value(ctx.cfg, "embed_text.truncate", True)),
            )
            if not emb_cfg.model:
                _log.warning("embed_text.model is empty. Skip image embedding.")
                return DocxIngestResult(
                    doc_id=doc_id,
                    doc_sha256=doc_sha,
                    doc_title=doc_title or doc_id,
                    source_uri=source_uri,
                    output_dir=str(output_dir),
                    md_path=str(md_cache_path),
                    provenance_path=str(prov_path),
                    images_dir=str(images_dir),
                    viewer_uri=viewer_uri,
                    text_index=ctx.os_text.index,
                    table_index=ctx.os_table.index,
                    tables_staging_index=ctx.os_tables_stage.index,
                    images_staging_index=ctx.os_images_stage.index,
                    chunk_count=chunk_count,
                    indexed_chunk_count=indexed_chunk_count,
                    staged_table_count=staged_table_count,
                    indexed_table_docs_count=indexed_table_docs_count,
                    indexed_table_rows_count=indexed_table_rows_count,
                    staged_image_count=staged_image_count,
                )
            emb_provider = OllamaEmbeddingProvider(emb_cfg)
            text_embedding_model = emb_cfg.model
            text_expected_dim = int(get_value(ctx.cfg, "embed_text.expected_dim", 4096))
            text_max_batch = int(get_value(ctx.cfg, "embed_text.max_batch_size", 32))

            img_url = get_value(ctx.cfg, "embed_image.ollama_base_url", None)
            if not img_url:
                img_url = get_value(ctx.cfg, "embed_image.base_url", "http://127.0.0.1:8088/embed")
            throttle = get_value(ctx.cfg, "embed_image.throttle_sec", None)
            if throttle is None:
                throttle = get_value(ctx.cfg, "embed_image.throtthle_sec", 0.0)

            image_embed_cfg = ImageEmbedConfig(
                url=str(img_url),
                timeout_sec=float(get_value(ctx.cfg, "embed_image.timeout_sec", 60.0)),
                expected_dim=int(get_value(ctx.cfg, "embed_image.expected_dim", 1024)),
                dimension=int(get_value(ctx.cfg, "embed_image.dimension", 1024)),
                max_images_per_request=int(get_value(ctx.cfg, "embed_image.max_images_per_request", 8)),
                retry_once=bool(get_value(ctx.cfg, "embed_image.retry_once", True)),
                throttle_sec=float(throttle or 0.0),
                model=str(get_value(ctx.cfg, "embed_image.model", "jinaai/jina-clip-v2")),
            )

            minio_cfg = MinIOConfig(
                endpoint=str(get_value(ctx.cfg, "minio.endpoint", "")),
                access_key=str(get_value(ctx.cfg, "minio.access_key", "")),
                secret_key=str(get_value(ctx.cfg, "minio.secret_key", "")),
                bucket=str(get_value(ctx.cfg, "minio.bucket", "")),
                secure=bool(get_value(ctx.cfg, "minio.secure", False)),
            )
            minio_reader = MinIOReader(minio_cfg)

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

            batch_size = int(get_value(ctx.cfg, "embed_image.batch_size", 16)) or 16
            buf: List[Dict[str, Any]] = []
            for hit in ctx.os_images_stage.scan(query=query, size=500):
                src = hit.get("_source", {})
                if not src:
                    continue
                buf.append(src)
                if len(buf) < batch_size:
                    continue
                pdf_ingest_mod._process_stage_batch(
                    buf=buf,
                    os_image=ctx.os_image,
                    os_images_stage=ctx.os_images_stage,
                    bulk_size=ctx.bulk_size,
                    emb_provider=emb_provider,
                    text_max_batch=text_max_batch,
                    text_expected_dim=text_expected_dim,
                    text_embedding_model=text_embedding_model,
                    image_embed_cfg=image_embed_cfg,
                    image_embedding_model=image_embed_cfg.model,
                    minio_reader=minio_reader,
                )
                buf = []
            if buf:
                pdf_ingest_mod._process_stage_batch(
                    buf=buf,
                    os_image=ctx.os_image,
                    os_images_stage=ctx.os_images_stage,
                    bulk_size=ctx.bulk_size,
                    emb_provider=emb_provider,
                    text_max_batch=text_max_batch,
                    text_expected_dim=text_expected_dim,
                    text_embedding_model=text_embedding_model,
                    image_embed_cfg=image_embed_cfg,
                    image_embedding_model=image_embed_cfg.model,
                    minio_reader=minio_reader,
                )

    return DocxIngestResult(
        doc_id=doc_id,
        doc_sha256=doc_sha,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        output_dir=str(output_dir),
        md_path=str(md_cache_path),
        provenance_path=str(prov_path),
        images_dir=str(images_dir),
        viewer_uri=viewer_uri,
        text_index=ctx.os_text.index,
        table_index=ctx.os_table.index,
        tables_staging_index=ctx.os_tables_stage.index,
        images_staging_index=ctx.os_images_stage.index,
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        staged_table_count=staged_table_count,
        indexed_table_docs_count=indexed_table_docs_count,
        indexed_table_rows_count=indexed_table_rows_count,
        staged_image_count=staged_image_count,
    )
