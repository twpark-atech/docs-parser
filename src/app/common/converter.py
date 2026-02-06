# ==============================================================================
# 목적 : DOCX to Markdown 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from app.parsing.regex import RE_SEP, RE_FILENAME


@dataclass(frozen=True)
class ProvenanceEntry:
    kind: str
    page_no: Optional[int]
    bbox: Optional[List[int]]
    loc_tokens: Optional[List[str]]
    ref: str
    text: Optional[str] = None
    image_id: Optional[str] = None
    image_rel_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportedImage:
    image_id: str
    rel_path: str
    page_no: Optional[int]
    bbox: Optional[List[int]]
    location_tokens: Optional[List[str]]


@dataclass(frozen=True)
class DocxMediaImage:
    name: str
    bytes: bytes
    ext: str


def _safe_int(x: Any) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return 0


def _bbox_to_list(bbox: Any) -> Optional[List[int]]:
    if bbox is None:
        return None

    for keys in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
        if all(hasattr(bbox, k) for k in keys):
            return [_safe_int(getattr(bbox, k)) for k in keys]

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [_safe_int(v) for v in bbox]

    if isinstance(bbox, dict):
        for keys in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
            if all(k in bbox for k in keys):
                return [_safe_int(bbox[k]) for k in keys]

    return None


def _get_page_no_and_bbox(item: Any) -> Tuple[Optional[int], Optional[List[int]]]:
    prov = getattr(item, "prov", None)
    if not prov:
        return None, None

    try:
        last = prov[-1]
    except Exception:
        return None, None

    page_no = getattr(last, "page_no", None)
    bbox = getattr(last, "bbox", None)

    try:
        pn = int(page_no) if page_no is not None else None
    except Exception:
        pn = None

    return pn, _bbox_to_list(bbox)


def _ref_block(ref: str, det: Optional[Union[str, List[int]]]) -> str:
    if det is None:
        return f"<|ref|>{ref}<|/ref|>"
    if isinstance(det, str):
        return f"<|ref|>{ref}<|/ref|><|det|>{det}<|/det|>"
    if isinstance(det, list) and len(det) == 4:
        return f"<|ref|>{ref}<|/ref|><|det|>[[{det[0]}, {det[1]}, {det[2]}, {det[3]}]]<|/det|>"
    return f"<|ref|>{ref}<|/ref|>"


def _normalize_iter_item(x: Any) -> Tuple[Optional[int], Any]:
    if isinstance(x, tuple) and len(x) == 2:
        a, b = x
        if isinstance(b, int):
            return b, a
        if isinstance(a, int):
            return a, b
    return None, x


def _iter_doc_items(doc: Any) -> Iterable[Tuple[Optional[int], Any]]:
    for attr in ("iterate_items", "iter_items"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            for x in fn():
                yield _normalize_iter_item(x)
            return

    body = getattr(doc, "body", None)
    if body is not None:
        children = getattr(body, "children", None) or getattr(body, "items", None)
        if children is not None:
            for x in children:
                yield None, x
            return


def _label_name(item: Any) -> str:
    lab = getattr(item, "label", None)
    if lab is None:
        return ""
    name = getattr(lab, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(lab).upper()


def _guess_kind(item: Any) -> str:
    lab = _label_name(item)
    if "TABLE" in lab:
        return "table"
    if "PICTURE" in lab or "IMAGE" in lab or "FIGURE" in lab:
        return "image"
    if "CAPTION" in lab:
        return "image_caption"
    if "SECTION_HEADER" in lab or "TITLE" in lab or "HEADING" in lab or "HEADER" in lab:
        return "sub_title"
    if "LIST_ITEM" in lab:
        return "list_item"

    name = item.__class__.__name__.lower()
    if "table" in name:
        return "table"
    if "picture" in name or "image" in name or "figure" in name:
        return "image"
    if "caption" in name:
        return "image_caption"
    if "title" in name or "heading" in name or "header" in name:
        return "sub_title"
    if "list" in name and "item" in name:
        return "list_item"
    return "text"


def _flatten_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        parts = [t for t in (_flatten_text(it) for it in x) if t]
        return " ".join(parts).strip()

    for key in ("text", "content", "value", "orig", "plain_text"):
        v = getattr(x, key, None)
        if v is not None:
            t = _flatten_text(v)
            if t:
                return t

    for m in ("export_to_text", "to_text", "get_text"):
        fn = getattr(x, m, None)
        if callable(fn):
            try:
                t = _flatten_text(fn())
                if t:
                    return t
            except Exception:
                pass

    try:
        return str(x).strip()
    except Exception:
        return ""


def _get_text(item: Any) -> str:
    for key in ("text", "content", "value", "orig"):
        t = _flatten_text(getattr(item, key, None))
        if t:
            return t
    return ""


def _get_table_html(item: Any) -> str:
    for m in ("export_to_html", "to_html"):
        fn = getattr(item, m, None)
        if callable(fn):
            try:
                s = fn()
                if isinstance(s, str) and s.strip():
                    return s
            except Exception:
                pass

    for key in ("raw_html", "html"):
        v = getattr(item, key, None)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _get_location_tokens(item: Any) -> Optional[List[str]]:
    fn = getattr(item, "get_location_tokens", None)
    if not callable(fn):
        return None
    try:
        toks = fn()
        if isinstance(toks, list):
            return [str(t) for t in toks if str(t)]
    except Exception:
        return None
    return None


def make_docx_image_id(*, doc_id: str, page_no: Optional[int], key: str) -> str:
    pn = page_no if page_no is not None else 1
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{pn}:img{h}"


def _sha1_bytes16(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def _save_docx_media_image_as_png(image_bytes: bytes, out_png: Path) -> None:
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes))
    img.save(out_png, format="PNG")


def normalize_table_html_to_tr_td(html: str, *, keep_br: bool = True) -> str:
    if not html or "<table" not in html.lower():
        return (html or "").replace("\r", "").replace("\n", "")

    def _norm_ws(s: str) -> str:
        s = (s or "").strip()
        return re.sub(r"\s+", " ", s)

    def _escape_with_br(s: str) -> str:
        if not keep_br:
            return escape(s)

        token = "__BR__"
        s = re.sub(r"<br\s*/?>", token, s, flags=re.I)
        parts = [escape(p) for p in s.split(token)]
        return "<br/>".join(parts)

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return html.replace("\r", "").replace("\n", "")

        out_rows: List[List[str]] = []
        for tr in table.find_all("tr"):
            row: List[str] = []
            for cell in tr.find_all(["td", "th"]):
                if keep_br:
                    for br in cell.find_all("br"):
                        br.replace_with("\n")

                text = cell.get_text("\n" if keep_br else " ", strip=True)

                if keep_br:
                    parts = [_norm_ws(x) for x in text.split("\n")]
                    parts = [p for p in parts if p]
                    text = "<br/>".join(parts)
                else:
                    text = _norm_ws(text)

                row.append(text)

            if row:
                out_rows.append(row)

        max_cols = max((len(r) for r in out_rows), default=0)
        if max_cols <= 0:
            return "<table></table>"

        out_rows = [r + [""] * (max_cols - len(r)) for r in out_rows]

        parts: List[str] = ["<table>"]
        for r in out_rows:
            tds = [f"<td>{_escape_with_br(c)}</td>" for c in r]
            parts.append("<tr>" + "".join(tds) + "</tr>")
        parts.append("</table>")

        return "".join(parts).replace("\r", "").replace("\n", "")

    except Exception:
        pass

    tr_blocks = re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    rows: List[List[str]] = []

    for tr in tr_blocks:
        td_blocks = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.I | re.S)
        row: List[str] = []
        for td in td_blocks:
            if keep_br:
                td = re.sub(r"<br\s*/?>", "\n", td, flags=re.I)

            text = re.sub(r"<[^>]+>", " ", td)
            if keep_br:
                parts = [_norm_ws(x) for x in text.split("\n")]
                parts = [p for p in parts if p]
                text = "<br/>".join(parts)
            else:
                text = _norm_ws(text)

            row.append(text)
        if row:
            rows.append(row)

    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 0:
        return "<table></table>"

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: List[str] = ["<table>"]
    for r in rows:
        tds = [f"<td>{_escape_with_br(c)}</td>" for c in r]
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</table>")

    return "".join(parts).replace("\r", "").replace("\n", "")


def export_grounded_md(
    doc: Any,
    *,
    doc_id: str,
    images_dir: Optional[Path] = None,
    images_subdir_name: str = "images",
    provenance_out: Optional[List[ProvenanceEntry]] = None,
    docx_media: Optional[List[DocxMediaImage]] = None,
) -> Tuple[str, List[ExportedImage]]:
    lines: List[str] = []
    exported: List[ExportedImage] = []

    media_iter = iter(docx_media or [])
    image_occ_idx = 0

    for page_hint, item in _iter_doc_items(doc):
        kind = _guess_kind(item)
        prov_page_no, bbox_det = _get_page_no_and_bbox(item)

        page_no: Optional[int] = prov_page_no
        if page_no is None and isinstance(page_hint, int):
            page_no = page_hint

        loc_tokens = _get_location_tokens(item)
        payload: List[str] = []

        if kind == "table":
            html = _get_table_html(item)
            if html:
                html = normalize_table_html_to_tr_td(html, keep_br=True)
                payload.append(html)
            else:
                txt = _get_text(item)
                if txt:
                    payload.append(txt)

            if provenance_out is not None and payload:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="table",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="table",
                        text=None,
                    )
                )

        elif kind == "image":
            image_occ_idx += 1

            try:
                media = next(media_iter)
            except StopIteration:
                media = None

            if media is not None:
                key = f"{image_occ_idx}:{media.name}:{_sha1_bytes16(media.bytes)}"
            else:
                self_ref = str(getattr(item, "self_ref", "") or "")
                key = f"{image_occ_idx}:{self_ref}"

            image_id = make_docx_image_id(doc_id=doc_id, page_no=page_no, key=key)

            rel_file = ""
            if images_dir is not None and media is not None:
                images_dir.mkdir(parents=True, exist_ok=True)

                safe_name = image_id.replace(":", "_")
                out_png = images_dir / f"{safe_name}.png"

                if not out_png.exists():
                    try:
                        _save_docx_media_image_as_png(media.bytes, out_png)
                    except Exception:
                        out_raw = images_dir / f"{safe_name}.bin"
                        if not out_raw.exists():
                            out_raw.write_bytes(media.bytes)

                if out_png.exists():
                    rel_file = f"{images_subdir_name}/{out_png.name}"
                else:
                    rel_file = f"{images_subdir_name}/{safe_name}.bin"

                exported.append(
                    ExportedImage(
                        image_id=image_id,
                        rel_path=rel_file,
                        page_no=page_no,
                        bbox=bbox_det,
                        location_tokens=loc_tokens,
                    )
                )

            lines.append(_ref_block("image", image_id))
            lines.append("")

            if provenance_out is not None:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="image",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="image",
                        text=None,
                        image_id=image_id,
                        image_rel_path=rel_file or None,
                    )
                )
            continue

        elif kind == "image_caption":
            txt = _get_text(item)
            if txt:
                payload.append(f"<center>[{txt}]</center>")

            if provenance_out is not None and txt:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="image_caption",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="image_caption",
                        text=txt,
                    )
                )

        elif kind == "sub_title":
            txt = _get_text(item)
            if txt:
                payload.append(txt if txt.lstrip().startswith("#") else f"## {txt}")

            if provenance_out is not None and txt:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="sub_title",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="sub_title",
                        text=txt,
                    )
                )

        elif kind == "list_item":
            txt = _get_text(item)
            if txt:
                payload.append(f"- {txt}")

            if provenance_out is not None and txt:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="list_item",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="list_item",
                        text=txt,
                    )
                )

        else:
            txt = _get_text(item)
            if txt:
                payload.append(txt)

            if provenance_out is not None and txt:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="text",
                        page_no=page_no,
                        bbox=bbox_det,
                        loc_tokens=loc_tokens,
                        ref="text",
                        text=txt,
                    )
                )

        if not payload:
            continue

        lines.append(_ref_block(kind, bbox_det))
        lines.extend(payload)
        lines.append("")

    md_text = "\n".join(lines).strip()
    return md_text, exported


def _split_md_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def md_table_block_to_html(
    md: str,
    *,
    escape_cells: bool = False,
    near_ref_only: bool = True,
    ref_window: int = 3,
) -> str:
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    def cell(x: str) -> str:
        return escape(x) if escape_cells else x

    def recent_has_table_ref(out_lines: list[str], *, window: int) -> bool:
        for j in range(1, min(window, len(out_lines)) + 1):
            if "<|ref|>table<|/ref|>" in out_lines[-j]:
                return True
        return False

    while i < len(lines):
        line = lines[i]
        is_table_ref = True if not near_ref_only else recent_has_table_ref(out, window=ref_window)

        if is_table_ref and (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and RE_SEP.match(lines[i + 1] or "")
        ):
            header = _split_md_row(lines[i])
            i += 2

            rows: List[List[str]] = []
            while i < len(lines):
                row_line = lines[i]
                if not row_line.strip().startswith("|"):
                    break
                rows.append(_split_md_row(row_line))
                i += 1

            col_n = len(header)
            html_parts = ["<table>"]
            html_parts.append("<tr>" + "".join(f"<td>{cell(h)}</td>" for h in header) + "</tr>")

            for r in rows:
                if len(r) < col_n:
                    r = r + [""] * (col_n - len(r))
                elif len(r) > col_n:
                    r = r[:col_n]
                html_parts.append("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>")

            html_parts.append("</table>")
            out.append("".join(html_parts))
            continue

        out.append(line)
        i += 1

    return "\n".join(out)
