# ==============================================================================
# 목적 : MD 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import List, Tuple

from app.parsing.regex import RE_PAGE_HEADER


def split_md_by_pages(md_text: str) -> List[Tuple[int, str]]:
    """Markdown 텍스트를 페이지 헤더 기준으로 (page_no, block) 리스트로 분할합니다.

    md_text에서 RE_PAGE_HEADER 정규식으로 페이지 헤더를 탐색합니다.

    Args:
        md_text: 페이지 헤더가 포함될 수 있는 Markdown 전체 문자열.

    Returns:
        (page_no, block) 튜플의 리스트.

    Raises:
        ValueError: 페이지 번호 그룹이 정수로 변환 불가할 경우.
    """
    if not md_text:
        return []
    
    matches = list(RE_PAGE_HEADER.finditer(md_text))
    if not matches:
        return [(0, md_text)]
    
    pages: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_no = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        block = md_text[start:end].strip("\n")
        pages.append((page_no, block))
    return pages