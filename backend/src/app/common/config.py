# ==============================================================================
# 목적 : 공통 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path

import yaml


def _candidate_paths(config_path: Path):
    yield config_path

    # Try alternate YAML extension for convenience.
    if config_path.suffix == ".yaml":
        yield config_path.with_suffix(".yml")
    elif config_path.suffix == ".yml":
        yield config_path.with_suffix(".yaml")

    # Resolve relative paths from project root as fallback.
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        root_path = project_root / config_path
        yield root_path
        if root_path.suffix == ".yaml":
            yield root_path.with_suffix(".yml")
        elif root_path.suffix == ".yml":
            yield root_path.with_suffix(".yaml")


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
    resolved = None
    for cand in _candidate_paths(config_path):
        if cand.exists():
            resolved = cand
            break

    if resolved is None:
        raise FileNotFoundError(f"config not found: {config_path}")

    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
