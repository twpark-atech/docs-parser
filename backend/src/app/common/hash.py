# ==============================================================================
# 목적 : Hash 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Bytes 데이터를 SHA-256으로 해싱해 16진수 문자열을 반환합니다.

    입력이 같으면 출력이 같습니다.
    출력은 항상 64자리의 HEX입니다.

    Args:
        data: 해싱할 원본 바이트.
        
    Returns:
        SHA-256 해시의 64자리, 16진수 문자열.
    """
    return hashlib.sha256(data).hexdigest()
    

def sha256_file(path: Path, chunks_size: int = 1024 * 1024) -> str:
    """파일을 청크 단위로 읽어 SHA-256으로 해싱해 16진수 문자열을 반환합니다.

    Args:
        path: 해시를 계산할 파일 경로.
        chunks_size: 한 번에 읽을 바이트 수.

    Returns:
        파일 전체 내용 기반 SHA-256 해시의 64자리 HEX 문자열.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunks_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_bytes16(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]