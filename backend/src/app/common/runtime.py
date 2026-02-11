# ==============================================================================
# 목적 : Runtime 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import time, logging, uuid
from datetime import datetime, timezone
from typing import Any, Optional

_log = logging.getLogger(__name__)


def request_with_retry(
    fn,
    *,
    retries: int = 5,
    base_sleep: float = 0.8,
    max_sleep: float = 8.0,
) -> Any:
    """요청 함수를 실행하고 실패 시 지수 백오프로 재시도합니다.

    fn() 실행 중 예외가 발생하면 최대 retries 횟수만큼 재시도합니다.
    재시도 간 대기 시간은 base_sleep * 2 ** attempt 형태로 증가하며, max_sleep을 상한으로 제한합니다.
    각 실패는 로깅 후 sleep을 수행하고 다음 시도를 진행합니다.
    모든 시도가 실패하면 마지막 예외를 원인으로 연결하여 예외를 발생시킵니다.

    Args:
        fn: 인자 없이 호출 가능한 요청 함수.
        retries: 총 시도 횟수(최초 시도 포함).
        base_sleep: 첫 재시도 대기 시간(초).
        max_sleep: 대기 시간 상항(초).

    Returns:
        fn()이 성공적으로 반환한 값.

    Raises:
        RuntimeError: retries 횟수만큼 시도했음에도 모두 실패한 경우.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep = min(max_sleep, base_sleep * (2 ** attempt))
            _log.warning("Request failed (attempt=%d/%d): %s -> sleep %.1fs", attempt + 1, retries, e, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Request failed after {retries} retries: {last_err}") from last_err


def now_utc() -> datetime:
    """UTC 타임존 기준 현재 시각을 반환합니다.
    
    datetime.now(timezone.utc)를 사용해 타임존 정보가 포함된 UTC 시간을 생성합니다.
    로그 타임스탬프, 파일 메타데이터, 분산 환경에서 시간 기준을 통일할 때 유용합니다.

    Returns:
        timezone.utc가 설정된 현재 datetime.
    """
    return datetime.now(timezone.utc)


def maybe_uuid(v: Any) -> Optional[uuid.UUID]:
    """입력 값을 UUID로 안전 변환합니다.
    
    입력이 None이면 None을 반환합니다.
    입력이 이미 uuid.UUID이면 그대로 반환합니다.
    입력이 문자열이면 uuid.UUID(...) 파싱을 시도하고, 실패하면 None을 반환합니다.
    그 외 타입은 변환하지 않고 None을 반환합니다.
    
    Args:
        v: UUID로 변환할 값.
        
    Returns:
        변환된 uuid.UUID 또는 변환 불가 시 None.    
    """
    if v is None:
        return None
    if isinstance(v, uuid.UUID):
        return v
    if isinstance(v, str):
        try:
            return uuid.UUID(v)
        except Exception:
            return None
    return None


def uuid5_from_chunk_id(chunk_id: str) -> uuid.UUID:
    """chunk_id로부터 결정적(UUIDv5) UUID를 생성합니다.
    
    UUIDv5는 (namespace, name) 입력이 같으면 항상 같은 UUID가 생성되는 결정적(deterministic) UUID입니다.
    chunk_id를 'name'으로 사용하면, 동일 chunk_id는 항상 동일 UUID로 매핑됩니다.
    이 UUID는 DB 키, 인덱스 문서 ID 등 "재생성 가능한 식별자"로 사용하기 적합합니다.

    Args:
        chunk_id: 청크 식별 문자열.

    Returns:
        chunk_id에 대응하는 UUIDv5.

    Raises:
        TypeError: uuid.uuid5 호출 인자가 올바르지 않을 때.
    """
    return uuid.uuid5(uuid.UUID("3d6f0a2e-3b22-4b4e-8ef0-5f9a7c1c2b11"), str(chunk_id))
