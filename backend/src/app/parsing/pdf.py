# ==============================================================================
# 목적 : PDF 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path
from typing import Union, Tuple, Any


def coerce_page_no_and_payload(
    item: Union[str, Path, bytes, bytearray, Tuple[int, Any]],
    fallback_page_no: int,
) -> Tuple[int, Any]:
    """입력 아이템을 (page_no, payload) 형태로 정규화합니다.

    item이 (page_no, payload) 형태의 길이 2 튜플이면 page_no를 int로 강제 변환해 반환합니다.
    그 외 타입이면 page_no는 fallback_page_no를 사용하고 payload는 item 자체를 반환합니다.

    Args:
        item: Page payload 또는 (page_no, payload) 튜플.
        fallback_page_no: item에 page_no가 없을 때 사용할 기본 페이지 번호.

    Returns:
        (page_no, payload) 튜플.

    Raises:
        ValueError: item이 튜플이고 첫 원소가 int로 변환 불가한 경우.   
    """
    if isinstance(item, tuple) and len(item) == 2:
        return int(item[0]), item[1]
    return fallback_page_no, item


def materialize_png_payload(payload: Any, *, out_dir: Path, page_no: int) -> Path:
    """PNG payload를 파일 경로(Path)로 물리화(Materialize)합니다.
    
    payload 타입에 따라 Path를 반환합니다.
    - Path: 그대로 반환
    - str: Path(str)로 변환해 반환
    - bytes/bytearray: out_dir 아래 "p{page_no:04d}.png"로 저장 후 경로 반환

    Args:
        payload: PNG payload.
        out_dir: bytes/bytesarray 저장 시 사용할 출력 디렉토리.
        page_no: 파일명 생성에 사용할 페이지 번호.

    Returns:
        PNG 파일 경로.

    Raises:
        TypeError: 지원하지 않는 payload 타입인 경우.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, Path):
        return payload
    if isinstance(payload, str):
        return Path(payload)
    if isinstance(payload, (bytes, bytearray)):
        png_path = out_dir / f"p{page_no:04d}.png"
        if not png_path.exists():
            png_path.write_bytes(bytes(payload))
        return png_path
    
    raise TypeError(f"Unsupported png payload type: {type(payload)!r}")