# ==============================================================================
# 목적 : 공통 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일을 로드하여 dict로 반환합니다.
    
    config_path의 YAML 파일을 PyYAML의 safe_load로 파싱합니다.
    YAML 내용이 비어있거나 파싱 결과가 falsy일 경우 빈 dict를 반환합니다.

    Args:
        config_path: YAML 설정 파일 경로.

    Returns:
        YAML을 dict로 파싱한 결과. 비어있으면 {} 반환.

    Raises:
        FileNotFoundError: config_path가 존재하지 않을 경우.
        yaml.YAMLError: YAML 문법 오류 등으로 파싱에 실패할 경우.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}