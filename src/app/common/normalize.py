# ==============================================================================
# 목적 : 정규화 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Dict, Tuple, Any, Optional


def ensure_text(d: Dict[str, Any]) -> str:
    """레코드(dict)에서 본문 텍스트를 우선순위 규칙으로 추출합니다.

    다양한 파서/모델 출력 포맷을 흡수하기 위해 여러 후보 키에서 텍스트를 탐색합니다.
    우선 순위는 ("text", "content", "markdown", "md") 순이며, 값이 비어있지 않은 문자열이면 즉시 반환합니다.
    위 키에서 찾지 못하면 "content"가 리스트인 경우, 각 원소가 {"text": "..."} 형태일 때 text만 모아 줄바꿈으로 합친 문자열을 반환합니다.
    어떤 경우에도 조건을 만족하지 않으면 빈 문자열("")을 반환합니다.

    Args:
        d: 텍스트를 포함할 수 있는 레코드 dict.

    Returns:
        추출된 텍스트.
    """
    for k in ("text", "content", "markdown", "md"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    content = d.get("content")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        if texts:
            return "\n".join(texts).strip()
    return ""


def ensure_page_range(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """레코드(dict)에서 페이지 범위를 (start, end)로 정규화해 반환합니다.

    문서 파싱 결과는 페이지 정보 키가 다양하게 존재할 수 있어, 본문 dict와 meta/metadata에서 다음 후보를 순차 탐색합니다.
    1) 단일 page: d["page"] 또는 meta["page"]가 int이면 (page, page) 반환.
    2) 범위 page_start/page_end 등 여러 후보 키 쌍을 찾아 (ps, pe) 반환.

    Args:
        d: 페이지 정보를 포함할 수 있는 레코드 dict.

    Returns:
        (page_start, page_end)
    """
    meta = d.get("meta") or d.get("metadata") or {}
    candidates = [
        ("page", "page"),
        ("page_start", "page_end"),
        ("pageStart", "pageEnd"),
        ("start_page", "end_page"),
    ]
    page = d.get("page") or meta.get("page")
    if isinstance(page, int):
        return page, page
    
    for a, b in candidates:
        ps = d.get(a) or meta.get(a)
        pe = d.get(b) or meta.get(b)
        if isinstance(ps, int) and isinstance(pe, int):
            return ps, pe
        
    return None, None


def ensure_order(idx: int, d: Dict[str, Any]) -> int:
    """레코드(dict)에서 순번(order)을 추출하고, 없으면 기본 idx를 반환합니다.

    다양한 스키마를 지원하기 위해 ("order", "idx", "index") 키를 순차 탐색합니다.
    해당 키들 중 int 값이 있으면 그 값을 반환하고, 모두 없으면 호출자가 넘긴 idx를 반환합니다.

    Args:
        idx: 레코드에 순번 정보가 없을 때 사용할 기본 순번.
        d: 순번 정보를 포함할 수 있는 레코드 dict.

    Returns:
        레코드의 순번(int).
    """
    for k in ("order", "idx", "index"):
        v = d.get(k)
        if isinstance(v, int):
            return v
    return idx