# ==============================================================================
# 목적 : Chunking 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import List, Dict, Any

import re

from app.parsing.md import split_md_by_pages
from app.parsing.regex import RE_IMG_TOKEN, RE_KIND_BBOX

_RE_IMAGE_ID = re.compile(r"\b[a-f0-9]{64}:p\d+:img[0-9a-f]{16}\b", re.IGNORECASE)


def _clean_line(line: str) -> str:
    if not line:
        return ""
    s = _RE_IMAGE_ID.sub("", line)
    s = RE_KIND_BBOX.sub("", s)
    return s.strip()


def _clean_md_text(md_text: str) -> str:
    if not md_text:
        return ""
    s = _RE_IMAGE_ID.sub("", md_text)
    s = RE_KIND_BBOX.sub("", s)
    return s


def build_chunks_from_md(
    doc_id: str,
    doc_title: str,
    source_uri: str,
    doc_sha: str,
    md_text: str,
    *,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 80,
) -> List[Dict[str, Any]]:
    """페이지 기반 Markdown 텍스트를 길이 제한에 맞는 청크 리스트로 변환합니다.

    md_text를 split_md_by_pages()로 (page_no, block) 단위로 분리합니다.
    빈 줄/공백 라인을 제거해 정규화합니다.
    너무 짧은 페이지는 스킵합니다.
    문단 단위로 나누고, max_chunk_chars를 넘지 않도록 문단을 누적합니다.
    누적 버퍼가 꽉 차면 청크를 확정하고 다음 문단부터 새 버퍼를 시작합니다.

    Args:
        doc_id: 문서 식별자.
        doc_title: 문서 제목.
        source_uri: 원본 위치 식별자.
        doc_sha: 문서 전체 내용 기반 SHA-256.
        md_text: Docling chunking 친화 포맷의 Markdown 전체 텍스트.
        max_chunk_chars: 한 청크에 허용할 최대 문자 수.
        min_chunk_chars: 청크로 인정할 최소 문자 수.

    Returns:
        청크 dict의 리스트.
        - doc_id: str
        - chunk_id: str
        - doc_title: str
        - source_uri: str
        - sha256: str
        - page_start: int
        - page_end: int
        - order: int
        - text: str
        - image_ids: list[str]
    """
    normalized: List[Dict[str, Any]] = []
    order = 0

    md_text = _clean_md_text(md_text)
    for page_no, block in split_md_by_pages(md_text):
        cleaned_lines = []
        for ln in block.splitlines():
            cleaned = _clean_line(ln)
            if cleaned:
                cleaned_lines.append(cleaned)
        text = "\n".join(cleaned_lines).strip()
        if len(text) < min_chunk_chars:
            continue

        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        buf = ""

        for p in paras:
            if not buf:
                buf = p
                continue
            if len(buf) + 2 + len(p) <= max_chunk_chars:
                buf = f"{buf}\n\n{p}"
            else:
                if len(buf) >= min_chunk_chars:
                    chunk_id = f"{doc_id}:p{page_no}-{page_no}:{order}"
                    normalized.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "source_uri": source_uri,
                        "sha256": doc_sha,
                        "page_start": page_no,
                        "page_end": page_no,
                        "order": order,
                        "text": buf,
                        "image_ids": RE_IMG_TOKEN.findall(buf),
                    })
                    order += 1
                buf = p
        
        if buf and len(buf) >= min_chunk_chars:
            chunk_id = f"{doc_id}:p{page_no}-{page_no}:{order}"
            normalized.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "source_uri": source_uri,
                "sha256": doc_sha,
                "page_start": page_no,
                "page_end": page_no,
                "order": order,
                "text": buf,
                "image_ids": RE_IMG_TOKEN.findall(buf),
            })
            order += 1

    return normalized
