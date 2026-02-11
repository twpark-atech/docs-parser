# ==============================================================================
# 목적 : 재시도 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-16
# AI 활용 여부 :
# ==============================================================================

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """재시도 정책 설정을 정의하는 클래스.
    재시도 관련 파라미터나 최대 시도 횟수를 설정합니다.

    Attributes:
        max_attempts: 최대 시도 횟수.
        backoff_sec: 재시도 전 대기 시간.
    """
    max_attempts: int = 2
    backoff_sec: float = 0.2


def run_with_retry(fn: Callable[[], T], *, policy: RetryPolicy, is_retriable: Callable[[Exception], bool]) -> T:
    """함수를 실행하고, 재시도 가능한 예외에 한해 정책에 따라 재시도합니다.

    fn을 실행하다 예외가 발생하면 is_retriable(e)로 재시도 여부를 판단합니다.
    재시도 가능하고 아직 max_attempts에 도달하지 않았다면 backoff_sec만큼 대기한 뒤 다시 시도합니다.
    재시도 불가 예외이거나 최대 시도 횟수에 도달하면 예외를 그대로 전파합니다.

    Args:
        fn: 인자 없이 호출 가능한 작업 함수.
        policy: 재시도 정책.
        is_retriable: 예외가 재시도 가능한지 판정하는 함수.

    Returns:
        fn이 성공적으로 반환한 값.

    Raises:
        Exception: fn 실행 중 발생한 예외를 그대로 전파합니다.
    """
    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= policy.max_attempts or not is_retriable(e):
                raise
            time.sleep(policy.backoff_sec)
    raise last_exc