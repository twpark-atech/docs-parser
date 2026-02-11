# ==============================================================================
# 목적 : 표 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import re
from typing import List, Tuple, Optional, Dict, Any

from bs4 import BeautifulSoup
from html import unescape

from app.common.hash import sha256_bytes
from app.parsing.regex import RE_HTML_TABLE

def normalize_table_html(raw_html: str) -> str:
    if not raw_html:
        return raw_html
    s = raw_html.strip()

    s = s.replace('\\"', '"').replace("\\'", "'")

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    s = unescape(s)
    return s


def extract_html_tables(md: str) -> List[str]:
    """Markdown 문자열에서 HTML <table>...</table> 블록을 추출합니다.

    RE_HTML_TABLE 정규식을 우선 적용해 매칭된 테이블들을 반환합니다.
    정규식 매칭이 없으면, fallback으로 "<table" 시작 위치를 찾습니다.
    각 시작점부터 다음 "<table" 시작 전까지를 후보 chunk로 잡은 뒤 내부에서 첫 "</table>" 닫힘 태그까지 잘라 테이블 블록을 복원합니다.
    
    Args:
        md: HTML table이 포함될 수 있는 Markdown 전체 문자열.

    Returns:
        추출된 HTML table 문자열 리스트.
    """
    if not md:
        return []
    
    tables = [m.group(1).strip() for m in RE_HTML_TABLE.finditer(md)]
    if tables:
        return tables
    
    lower = md.lower()
    starts = [m.start() for m in re.finditer(r"<table\b", lower)]
    if not starts:
        return []
    
    out: List[str] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(md)
        chunk = md[s:e]
        close = chunk.lower().find("</table>")
        if close != -1:
            chunk = chunk[: close + len("</table>")]
        out.append(chunk.strip())
    return out


def parse_table(html: str) -> Tuple[List[str], List[List[str]]]:
    """HTML table로 파싱하여 header와 rows(2D 배열)를 반환합니다.

    BeautifulSoup로 <tr> 목록을 읽어 첫 번째 행을 header로 간주합니다.
    첫 행에서 <td> 텍스트를 우선 header로 사용하고, <td>가 없으면 <th> 텍스트를 header로 사용합니다.
    header가 비어 있거나 컬럼 수가 0이면 ([], [])를 반환합니다.

    rowspan>1인 셀은 spans에 저장해 다음 행들에 동일 텍스트를 자동 채워 넣습니다.
    
    Args:
        html: <table>...</table> 형태의 HTML 문자열

    Returns:
        (header, rows)
    """
    soup = BeautifulSoup(html, "html.parser")
    trs = soup.find_all("tr")
    if not trs:
        return [], []
    
    header = [td.get_text("\n", strip=True) for td in trs[0].find_all("td")]
    if not header:
        header = [th.get_text("\n", strip=True) for th in trs[0].find_all("th")]
    
    n = len(header)
    if n == 0:
        return [], []
    
    spans: List[Optional[Dict[str, Any]]] = [None] * n
    rows: List[List[str]] = []

    for tr in trs[1:]:
        row: List[Optional[str]] = [None] * n
        
        for i in range(n):
            if spans[i]:
                row[i] = spans[i]["text"]
                spans[i]["remain"] -= 1
                if spans[i]["remain"] == 0:
                    spans[i] = None

        col = 0
        tds = tr.find_all("td")
        if not tds:
            tds = tr.find_all("th")

        for td in tds:
            while col < n and row[col] is not None:
                col += 1
            if col >= n:
                break

            text = td.get_text("\n", strip=True)
            row[col] = text

            rs = int(td.get("rowspan", 1))
            if rs > 1:
                spans[col] = {"text": text, "remain": rs - 1}

            col += 1

        rows.append([(x or "") for x in row])

    return header, rows


def build_table_text(header: List[str], rows: List[List[str]], max_rows: int = 50) -> str:
    """테이블을 검색/임베딩 친화적인 텍스트로 직렬화합니다.
    
    첫 줄: "H1:<col1> | H2:<col2> | ..." 형태의 헤더 라인
    다음 줄들: 각 행을 dict로 매핑해 "R{idx}: colA: vA | colB: vB | ..." 형태로 출력
    rows가 max_rows를 초과하면 마지막에 "...(rows truncated: N total)" 라인을 추가

    Args:
        header: 컬럼명 리스트.
        rows: 테이블 행 2D 리스트.
        max_rows: 직렬화할 최대 행 수.

    Returns:
        직렬화된 텍스트.
    """
    if not header:
        return ""
    
    lines: List[str] = []
    lines.append(" | ".join([f"H{i+1}:{h}" for i, h in enumerate(header)]))
    for r_idx, r in enumerate(rows[: max_rows]):
        row_obj = {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
        row_text = " | ".join([f"{k}: {v}" for k, v in row_obj.items()])
        lines.append(f"R{r_idx}: {row_text}")
    if len(rows) > max_rows:
        lines.append(f"...(rows truncated: {len(rows)} total)")
    return "\n".join(lines)


def build_row_text(header: List[str], row: List[str]) -> Tuple[Dict[str, str], str]:
    """단일 행을 (row_obj, row_text)로 변환합니다.
    
    row_obj는 header -> value 매핑 dict입니다.
    row_text는 "colA: vA | colB: vB | ..." 형태로 직렬화된 문자열입니다.
    row 길이가 header보다 짧으면 누락된 값은 ""로 채웁니다.

    Args:
        header: 컬럼명 리스트.
        row: 단일 행 값 리스트.

    Returns:
        (row_obj, row_text)
    """
    row_obj = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
    row_text = " | ".join([f"{k}: {v}" for k, v in row_obj.items()])
    return row_obj, row_text


def build_table_id(doc_id: str, page_no: int, order: int, raw_html: str) -> Tuple[str, str]:
    """테이블 HTML로부터 (table_id, table_sha256)를 생성합니다.
    
    raw_html을 UTF-8 바이트로 인코딩(errors="ignore")한 뒤 sha256_bytes로 해시를 계산합니다.
    table_id 포맷: "{doc_id}:p{page_no:04d}:t{order:02d}:{sha12}"

    Args:
        doc_id: 문서 식별자.
        page_no: 테이블이 위치한 페이지 번호.
        order: 페이지 내 테이블 순번.
        raw_html: 테이블 원본 HTML 문자열.

    Returns:
        (table_id, table_sha256)
    """
    raw_bytes = raw_html.encode("utf-8", errors="ignore")
    table_sha = sha256_bytes(raw_bytes)
    table_id = f"{doc_id}:p{int(page_no):04d}:t{int(order):02d}:{table_sha[:12]}"
    return table_id, table_sha