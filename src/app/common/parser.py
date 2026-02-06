# ==============================================================================
# 목적 : Parse 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path

try:
    import pymupdf as fitz
except Exception:
    import fitz


def get_value(cfg: dict, path: str, default=None):
    """중첩 dict에서 dotted path로 값을 안전하게 조회합니다.

    path를 '.'를 기준으로 분리한 키 시퀀스를 따라가며 중첩 dict 값을 조회합니다.
    탐색 중 현재 값이 dict가 아니거나, 키가 존재하지 않으면 default를 반환합니다.
    
    Args:
        cfg: 조회 대상 중첩 dict.
        path: '.'으로 구분된 키 경로 문자열.
        default: 경로가 유효하지 않거나 키가 없을 때 반환할 기본값.

    Returns:
        dotted path로 조회된 값.
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pdf_to_page_pngs(pdf_path: Path, scale: float = 2.0) -> list[bytes]:
    r"""PDF를 페이지별 PNG bytes로 렌더링합니다.

    PyMuPDF(fitz)를 사용해 PDF의 각 페이지를 렌더링하고, PNG 포맷의 bytes로 반환합니다.
    scale은 렌더링 해상도에 영향을 주며(확대 비율), 값이 클수록 이미지가 커지고 처리 비용이 증가합니다.
    
    Args:
        pdf_path: 입력 PDF 파일 경로.
        scale: 렌더링 확대 비율.

    Returns:
        PDF 페이지 순서대로 정렬된 PNG bytes 리스트.

    Raises:
        fitz.FileDataError: PDF 파일이 손상되었거나 읽을 수 없는 경우. 
        RuntimeError: PyMuPDF 내부 렌더링 실패 등.
        OSError: 파일 접근/읽기 과정에서 발생하는 OS 레벨 오류.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(scale, scale)
    pages = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def build_md_from_pages(page_texts: list[str]) -> str:
    """페이지별 OCR 텍스트를 Docling Chunking 친화적인 Markdown로 결합합니다.

    각 페이지의 텍스트를 '## Page N' 헤더로 구분하고, 페이지 사이에 '---' 구분선을 삽입합니다.
    각 페이지 텍스트는 strip() 처리 후 삽입하며, 결과 문자열은 마지막에 개행('\\n')으로 끝나도록 반환합니다.
    
    Args:
        page_texts: 페이지별 OCR 결과 텍스트 리스트.

    Returns:
        결합된 Markdown 문자열.
    """
    parts = []
    for i, txt in enumerate(page_texts, start=1):
        parts.append(f"## Page {i}\n\n{txt.strip()}\n")
        parts.append("\n---\n")
    return "\n".join(parts).strip() + "\n"
