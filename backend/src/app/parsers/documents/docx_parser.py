import hashlib
import io
import html
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.common.converter import DocxMediaImage, ExportedImage, ProvenanceEntry
from app.common.hash import sha1_bytes16, sha256_file
from app.common.parser import get_value
from app.pipelines.document_ingest import build_context
from app.parsing.regex import A_NS, REL_NS, R_NS, W_NS

_log = logging.getLogger(__name__)


class DocxParser:
    @staticmethod
    def _source_name_for_object_key(source_name: str, default_ext: str) -> tuple[str, str]:
        raw = (source_name or "").strip()
        base = re.split(r"[\\/]", raw)[-1].strip() if raw else ""
        if not base:
            base = f"document{default_ext}"
        stem = Path(base).stem.strip() or "document"
        ext = Path(base).suffix.lstrip(".").strip() or default_ext.lstrip(".")
        return f"{stem}_{ext}", base

    @staticmethod
    def parse(
        *,
        config_path: Optional[Path] = None,
        input_docx: Optional[Path] = None,
        doc_id_override: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        ctx = build_context(config_path=config_path, source_type="docx")
        cfg = ctx.cfg
        output_dir = ctx.output_dir

        if input_docx is None:
            cfg_path = get_value(cfg, "paths.data_folder", None)
            cfg_docx = get_value(cfg, "paths.input_docx", None)
            if cfg_docx:
                input_docx = Path(str(cfg_path)) / cfg_docx

        if input_docx is None:
            raise ValueError("input_docx is required. (set paths.input_docx in config or pass input_docx)")

        input_doc_path = Path(input_docx)
        if not input_doc_path.exists():
            raise FileNotFoundError(f"DOCX not found: {input_doc_path}")

        doc_sha = sha256_file(input_doc_path)
        doc_id = str(doc_id_override).strip() if doc_id_override else doc_sha
        source_filename = source_name or input_doc_path.name
        minio_prefix, minio_original_name = DocxParser._source_name_for_object_key(source_filename, ".docx")
        original_key = f"{minio_prefix}/{minio_original_name}"
        original_upload: Dict[str, Any] = {}
        try:
            original_upload = ctx.minio_writer.upload_file_to_key(
                str(input_doc_path),
                object_key=original_key,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as e:
            _log.warning("MinIO original upload failed: %s", e)

        provenance: List[ProvenanceEntry] = []
        tables_out: List[Dict[str, str]] = []
        md_text = DocxParser.export_md_from_docx(
            docx_path=input_doc_path,
            doc_id=doc_id,
            provenance_out=provenance,
            tables_out=tables_out,
        )

        docx_media = DocxParser.extract_docx_media(input_doc_path)
        DocxParser.save_all_docx_media_images(
            doc_id=doc_id,
            docx_media=docx_media,
            minio_writer=ctx.minio_writer,
            minio_prefix=minio_prefix,
        )
        DocxParser.save_all_docx_tables(
            tables=tables_out,
            minio_writer=ctx.minio_writer,
            minio_prefix=minio_prefix,
        )
        title = Path(source_filename).stem
        minio_bucket = str(get_value(cfg, "minio.bucket", ""))
        return {
            "doc_id": doc_id,
            "title": title,
            "source_name": source_filename,
            "source_uri": source_filename,
            "viewer_uri": f"minio://{minio_bucket}/{original_key}" if minio_bucket else "",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "size_bytes": int(input_doc_path.stat().st_size),
            "minio_bucket": minio_bucket,
            "minio_key": original_key,
            "minio_etag": str(original_upload.get("etag") or ""),
            "paragraphs": DocxParser._split_paragraphs(md_text),
        }

    @staticmethod
    def export_md_from_docx(
        *,
        docx_path: Path,
        doc_id: str,
        provenance_out: Optional[List[ProvenanceEntry]] = None,
        tables_out: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        xml = DocxParser._open_docx_document_xml(docx_path)
        root = ET.fromstring(xml)
        rid_to_image_id = DocxParser._rid_to_image_id(docx_path, doc_id)
        body = root.find(".//w:body", W_NS)
        if body is None:
            return ""

        lines: List[str] = []
        table_order = 0
        for child in list(body):
            tag = child.tag
            if tag.endswith("}p"):
                text = DocxParser._paragraph_text(child, rid_to_image_id, doc_id)
                if text:
                    lines.append(text)
                    lines.append("")
            elif tag.endswith("}tbl"):
                table_html = DocxParser._table_html(child)
                if table_html:
                    table_order += 1
                    table_id = DocxParser._table_id(doc_id, table_order, table_html)
                    lines.append(f"[TABLE_ID:{table_id}]")
                    lines.append("")
                    if tables_out is not None:
                        tables_out.append(
                            {
                                "table_id": table_id,
                                "table_md": table_html,
                            }
                        )
                    if provenance_out is not None:
                        provenance_out.append(
                            ProvenanceEntry(
                                kind="table",
                                page_no=None,
                                bbox=None,
                                loc_tokens=None,
                                ref="table",
                                text=None,
                                image_id=table_id,
                                image_rel_path=None,
                            )
                        )
        return "\n".join(lines).strip()

    @staticmethod
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

    @staticmethod
    def save_all_docx_media_images(
        *,
        doc_id: str,
        docx_media: List[DocxMediaImage],
        minio_writer=None,
        minio_prefix: str = "",
    ) -> List[ExportedImage]:
        exported: List[ExportedImage] = []
        for m in docx_media:
            key = sha1_bytes16(m.bytes)
            image_id = DocxParser._image_id(doc_id=doc_id, key=key)

            safe_name = image_id.replace(":", "_")
            rel_path = ""
            png_bytes = DocxParser._to_png_bytes(m.bytes)
            if png_bytes is not None:
                rel_path = f"images/{safe_name}.png"
                if minio_writer is not None and minio_prefix:
                    try:
                        minio_writer.upload_bytes_to_key(
                            png_bytes,
                            object_key=f"{minio_prefix}/{rel_path}",
                            content_type="image/png",
                        )
                    except Exception as e:
                        _log.warning("MinIO image upload failed: %s", e)
            else:
                rel_path = f"images/{safe_name}.bin"
                if minio_writer is not None and minio_prefix:
                    try:
                        minio_writer.upload_bytes_to_key(
                            m.bytes,
                            object_key=f"{minio_prefix}/{rel_path}",
                            content_type="application/octet-stream",
                        )
                    except Exception as e:
                        _log.warning("MinIO image upload failed: %s", e)
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

    @staticmethod
    def save_all_docx_tables(
        *,
        tables: List[Dict[str, str]],
        minio_writer=None,
        minio_prefix: str = "",
    ) -> None:
        for t in tables:
            table_id = str(t.get("table_id") or "").strip()
            table_md = str(t.get("table_md") or "").strip()
            if not table_id:
                continue
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", table_id)
            file_name = f"{safe_name}.md"
            payload = (table_md + ("\n" if table_md else "")).encode("utf-8")
            if minio_writer is not None and minio_prefix:
                try:
                    minio_writer.upload_bytes_to_key(
                        payload,
                        object_key=f"{minio_prefix}/tables/{file_name}",
                        content_type="text/markdown; charset=utf-8",
                    )
                except Exception as e:
                    _log.warning("MinIO table upload failed: %s", e)

    @staticmethod
    def _open_docx_document_xml(docx_path: Path) -> str:
        with zipfile.ZipFile(docx_path, "r") as zf:
            return zf.read("word/document.xml").decode("utf-8", errors="replace")

    @staticmethod
    def _image_id(doc_id: str, key: str) -> str:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return f"{doc_id}:img{h}"

    @staticmethod
    def _table_id(doc_id: str, order: int, table_html: str) -> str:
        h = hashlib.sha1(table_html.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}:tbl{order:04d}:{h}"

    @staticmethod
    def _rid_to_image_id(docx_path: Path, doc_id: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        with zipfile.ZipFile(docx_path, "r") as zf:
            try:
                rels_xml = zf.read("word/_rels/document.xml.rels").decode("utf-8", errors="replace")
            except KeyError:
                return out

            rels_root = ET.fromstring(rels_xml)
            for rel in rels_root.findall("./pr:Relationship", REL_NS):
                rid = rel.attrib.get("Id")
                rel_type = rel.attrib.get("Type", "")
                target = rel.attrib.get("Target", "")
                if not rid or "/image" not in rel_type or not target:
                    continue

                media_path = target.lstrip("/")
                if not media_path.startswith("word/"):
                    media_path = f"word/{media_path}"
                try:
                    image_bytes = zf.read(media_path)
                    key = hashlib.sha1(image_bytes).hexdigest()
                except Exception:
                    key = target
                out[rid] = DocxParser._image_id(doc_id, key)
        return out

    @staticmethod
    def _paragraph_text(p: ET.Element, rid_to_image_id: Dict[str, str], doc_id: str) -> str:
        parts: List[str] = []
        for r in p.findall("./w:r", W_NS):
            for node in list(r):
                tag = node.tag
                if tag.endswith("}t") and node.text:
                    parts.append(node.text)
                elif tag.endswith("}tab"):
                    parts.append("\t")
                elif tag.endswith("}br"):
                    parts.append("\n")
                elif tag.endswith("}drawing"):
                    if not doc_id:
                        continue
                    for blip in node.findall(".//a:blip", A_NS):
                        rid = blip.attrib.get(f"{{{R_NS}}}embed")
                        if not rid:
                            continue
                        image_id = rid_to_image_id.get(rid) or DocxParser._image_id(doc_id, rid)
                        parts.append(f"[{image_id}]")
        return "".join(parts).strip()

    @staticmethod
    def _table_html(tbl: ET.Element) -> str:
        rows: List[List[str]] = []
        for tr in tbl.findall("./w:tr", W_NS):
            row: List[str] = []
            for tc in tr.findall("./w:tc", W_NS):
                paras = tc.findall(".//w:p", W_NS)
                cell_lines = []
                for p in paras:
                    t = DocxParser._paragraph_text(p, {}, doc_id="")
                    if t:
                        cell_lines.append(t)
                raw_cell = "<br/>".join(cell_lines).strip()
                escaped = html.escape(raw_cell, quote=False)
                row.append(escaped.replace("&lt;br/&gt;", "<br/>"))
            if row:
                rows.append(row)

        if not rows:
            return ""

        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        out = ["<table>"]
        for r in rows:
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        out.append("</table>")
        return "".join(out)

    @staticmethod
    def _save_docx_media_image_as_png(image_bytes: bytes, out_png: Path) -> None:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img.save(out_png, format="PNG")

    @staticmethod
    def _to_png_bytes(image_bytes: bytes) -> Optional[bytes]:
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(image_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def _split_paragraphs(md_text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n\s*\n+", md_text or "") if p.strip()]
