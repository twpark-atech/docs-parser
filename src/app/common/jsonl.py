# ==============================================================================
# 목적 : JSONL 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import json
from pathlib import Path
from typing import Any, Mapping


def append_jsonl(records: Any, jsonl_path: Path) -> None:
    """레코드를 JSONL 파일에 한 줄로 추가합니다.
    
    단일 dict이면 리스트로 감싸 다건 처리로 통일합니다.
    상위 디렉토리는 자동으로 생성합니다.
    "UTF-8"과 "ensure_ascii=False"로 저장됩니다.

    Args:
        records: 단일 레코드 또는 레코드 리스트.
        jsonl_path: 출력 jsonl 파일 경로.

    Returns:
        jsonl 파일로 저장.

    Raises:
        TypeError: JSON 직렬화 불가 타입이 포함될 경우.
        OSError: 파일 I/O에 오류가 발생할 경우
    """
    if not records:
        return
    if isinstance(records, Mapping):
        records = [records]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    """JSONL 파일의 비어있지 않은 라인 수를 반환합니다.

    파일이 존재하지 않으면 0을 반환합니다.
    "UTF-8"로 읽고, strip() 후 비어있지 않은 줄만 카운트합니다.
    
    Args:
        path: JSONL 파일 경로.

    Returns:
        비어있지 않은 줄 수.
    """
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n