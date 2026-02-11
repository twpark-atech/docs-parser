# ==============================================================================
# 목적 : Batch 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Sequence, Any, Iterable, Tuple


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int, Sequence[Any]]]:
    """주어진 시퀀스를 배치 크기 단위로 분할하여 순회합니다.

    items를 batch_size 단위로 잘라서 (start, end, batch_items)를 생성합니다.
    여기서 end는 python slicing rule과 동일하게 '미포함(end-exclusive)' 인덱스입니다.
    
    Args:
        items: 배치로 분리할 시퀀스 데이터.
        batch_size: 한 배치에 포함할 아이템 개수.

    Returns:
        (start, end, batch_items)의 iterator.

    Raises:
        ValueError: batch size가 0 이하일 경우.     
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = len(items)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, items[start:end]