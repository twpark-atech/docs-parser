# ==============================================================================
# 목적 : ID 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Optional, Tuple


def chunk_id_of(doc_id: str, page_start: Optional[int], page_end: Optional[int], order: int) -> str:
    """문서 내 청크를 식별하는 결정적 ID 문자열을 생성합니다.

    page_start가 None이면 0, page_end가 None이면 page_start로 보정합니다.
    포맷은 "{doc_id}:p{ps}-{pe}:{order}"로 고정됩니다.

    Args:
        doc_id: 문서 식별자(파일 해시 등).
        page_start: 시작 페이지(없으면 0으로 처리).
        page_end: 끝 페이지(없으면 시작 페이지와 동일).
        order: 동일 페이지 범위 내 청크 순번.

    Returns:
        청크 ID 문자열.
    
    """
    ps = page_start if page_start is not None else 0
    pe = page_end if page_end is not None else ps
    return f"{doc_id}:p{ps}-{pe}:{order}"


def s3_uri(bucket: str, key: str) -> str:
    """Bucket/Key로 S3 스타일 URI을 생성합니다.

    MinIO(S3 호환) 객체 위치를 문자열로 표준화합니다.
    Key가 경로를 포함해도 그대로 유지됩니다.

    Args:
        bucket: 버킷 이름.
        key: 객체 키.
    
    Returns:
        "s3://{bucket}/{key}"의 문자열.
    """
    return f"s3://{bucket}/{key}"


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """S3 URI를 (bucket, key)로 파싱합니다.

    s3:// 스킴을 강제합니다.
    bucket/key 누락을 예외로 처리합니다.
    'split("/", 1)'로 key 내 "/"를 보존합니다.

    Args:
        uri: "s3://bucket/key" 형식의 문자열.

    Returns:
        버킷 이름, 객체 키.

    Raises:
        ValueError: 스킴이 일치하지 않는 경우, Key가 누락된 경우, bucket/key가 비어있는 경우.
    """
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"invalid s3 uri: {uri}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"invalid s3 uri (missing key): {uri}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri: {uri}")
    return bucket, key