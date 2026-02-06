# ==============================================================================
# 목적 : Embedding 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import time, base64
from dataclasses import dataclass
from typing import List, Optional, Callable, TypeVar

import requests

from app.common.retry import RetryPolicy, run_with_retry
from app.storage.embedding import OllamaEmbeddingProvider

T = TypeVar("T")


def embed_texts(
    emb_provider: OllamaEmbeddingProvider,
    texts: List[str],
    *,
    max_batch_size: int,
    expected_dim: int,
) -> List[List[float]]:
    """텍스트 리스트를 배치로 임베딩하고, 결과의 개수/차원을 검증합니다.
    
    text를 max_batch_size 단위로 분할하여 emb_provider.embed(batch)를 반복 호출합니다.
    모든 배치를 합친 뒤 최종적으로 벡터 개수가 입력 텍스트 개수와 동일한지 검사합니다.

    Args:
        emb_provider: 텍스트 임베딩 제공자.
        texts: 임베딩할 텍스트 리스트.
        max_batch_size: provider 호출 시 한 번에 보낼 최대 텍스트 개수.
        expected_dim: 기대하는 임베딩 벡터 차원.

    Returns:
        texts와 동일한 순서를 갖는 임베딩 벡터 리스트.

    Raises:
        RuntimeError: provider가 빈 벡터를 반환하거나, 벡터 차원이 기대 벡터 차원과 다르거나, 최종 벡터 개수가 입력 개수와 다른 경우.
    """
    vectors: List[List[float]] = []
    for start in range(0, len(texts), max_batch_size):
        batch = texts[start : start + max_batch_size]
        batch_vecs = emb_provider.embed(batch)
        if not batch_vecs:
            raise RuntimeError("Embedding provider returned empty vectors.")
        for v in batch_vecs:
            if len(v) != expected_dim:
                raise RuntimeError(f"Unexpected embedding dim: {len(v)} (expected {expected_dim})")
        vectors.extend(batch_vecs)

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(texts)}")
    return vectors


def embed_text(
    emb_provider: OllamaEmbeddingProvider,
    text: str,
    *,
    expected_dim: int,
) -> List[float]:
    """단일 텍스트를 임베딩하고 결과 차원을 검증합니다.
    
    embed_texts를 이용해 text 1건을 임베딩합니다.
    반환 벡터의 차원은 기대 벡터 차원을 만족해야 합니다.
    
    Args:
        emb_provider: 텍스트 임베딩 제공자.
        text: 임베딩할 단일 텍스트.
        expected_dim: 기대 임베딩 차원.

    Returns:
        단일 임베딩 벡터.

    Raises:
        RuntimeError: provider 결과가 비어있거나 차원/개수 검증에 실패한 경우.
    """
    vecs = embed_texts(
        emb_provider,
        [text],
        max_batch_size=1,
        expected_dim=expected_dim,
    )
    return vecs[0]


@dataclass(frozen=True)
class ImageEmbedConfig:
    """이미지 임베딩 HTTP 호출 설정을 정의하는 클래스.
    이미지 바이트를 base64로 인코딩하여 cfg.url 엔드포인트로 POST 요청을 보냅니다.
    요청/응답 처리에서 기대 벡터 차원과 1회 요청당 최대 이미지 수, 실패 시 1회 재시도 여부, 요청 간 지연 등을 설정합니다.
    
    Attributes:
        url: 임베딩 API 엔드포인트 URL.
        timeout_sec: HTTP 요청 타임아웃(초).
        expected_dim: 응답 벡터의 기대 차원(검증용).
        dimension: 요청 payload에 포함할 dimension 값.
        max_images_per_request: 한 번의 요청에 포함할 최대 이미지 개수.
        retry_once: 요청 실패 시 동일 파트를 1회 재시도할지 여부.
        throttle_sec: 각 요청 전 대기 시간(초).
        model: 사용 모델 식별자.
    """
    url: str = "http://127.0.0.1:8008/embed"
    timeout_sec: float = 60.0
    expected_dim: int = 1024
    dimension: int = 1024
    max_images_per_request: int = 8
    retry_once: bool = True
    throttle_sec: float = 0.0
    model: str = "jinaai/jina-clip-v2"


def _to_b64(image_bytes: bytes) -> str:
    """이미지 파이트를 base64(ASCII 문자열)로 인코딩합니다.
    
    Args:
        image_bytes: 원본 이미지 바이트.
        
    Returns:
        base64로 인코딩된 ASCII 문자열.
    """
    return base64.b64encode(image_bytes).decode("ascii")


def _call_embed_once(*, cfg: ImageEmbedConfig, part: List[bytes]) -> List[List[float]]:
    """이미지 임베딩 API를 1회 호출하고 결과를 검증합니다.
    
    part의 각 이미지 바이트를 base64로 인코딩하여 POST 요청을 수행합니다.
    응답 JSON의 "vectors" 필드에서 벡터 리스트를 가져오며, 벡터 개수와 벡터 차원을 검증합니다.

    Args:
        cfg: 이미지 임베딩 설정.
        part: 한 번의 요청에 포함할 이미지 바이트 리스트.

    Returns:
        part와 동일한 순서의 임베딩 벡터 리스트.

    Raises:
        requests.HTTPError: HTTP status가 4xx/5xx일 경우.
        RuntimeError: 백터 개수 또는 차원이 일치하지 않을 경우.
        ValueError: r.json() 파싱에 실패하거나 응답이 JSON이 아닐 경우.
    """
    payload = {
        "images_b64": [_to_b64(b) for b in part],
        "dimension": int(cfg.dimension or cfg.expected_dim),
    }
    r = requests.post(cfg.url, json=payload, timeout=cfg.timeout_sec)
    r.raise_for_status()
    data = r.json()

    vecs = data.get("vectors", [])
    if len(vecs) != len(part):
        raise RuntimeError(f"vector count mismatch: {len(vecs)} != {len(part)}")
    
    for v in vecs:
        if len(v) != cfg.expected_dim:
            raise RuntimeError(f"Unexpected image embedding dim: {len(v)} (expected {cfg.expected_dim})")
    
    return vecs


def embed_images_bytes_batch(
    images: List[bytes],
    *,
    cfg: ImageEmbedConfig,
) -> List[List[float]]:
    """이미지 바이트 리스트를 배치로 임베딩하고, 필요 시  1회 재시도합니다.
    
    image가 비어있으면 빈 리스트를 반환합니다.
    cfg.max_images_per_request 단위로 이미지를 분할하여 _cdall_embed_once를 호출합니다.
    최종적으로 반환 벡터 개수가 입력 이미지 개수와 동일한지 검증합니다.

    Args:
        images: 임베딩할 이미지 바이트 리스트.
        cfg: 이미지 임베딩 설정.

    Returns:
        image와 동일한 순서를 갖는 임베딩 벡터 리스트.

    Raises:
        RuntimeError: 최종 벡터 개수가 일치하지 않거나 _call_embed_once 검증에 실패한 경우.
        requests.HTTPError / requests.RequestExceiption: HTTP 요청에 실패한 경우.
    """
    if not images:
        return []
    
    out: List[List[float]] = []

    step = max(1, int(cfg.max_images_per_request))
    for start in range(0, len(images), step):
        part = images[start : start + step]

        try:
            if cfg.throttle_sec > 0:
                time.sleep(cfg.throttle_sec)
            out.extend(_call_embed_once(cfg=cfg, part=part))
        except Exception:
            if not cfg.retry_once:
                raise
            if cfg.throttle_sec > 0:
                time.sleep(cfg.throttle_sec)
            out.extend(_call_embed_once(cfg=cfg, part=part))

    if len(out) != len(images):
        raise RuntimeError(f"embedding count mismatch: {len(out)} != {len(images)}")

    return out


def embed_image_bytes(image_bytes: bytes, *, cfg: ImageEmbedConfig) -> List[float]:
    """단일 이미지 바이트를 임베딩합니다.
    
    embed_image_bytes_batch를 이용해 이미지 1건을 임베딩하고 첫 번째 벡터를 반환합니다.
    
    Args:
        image_bytes: 임베딩할 단일 이미지 바이트.
        cfg: 이미지 임베딩 설정.

    Returns:
        단일 이미지 임베딩 벡터.
    """
    return embed_images_bytes_batch([image_bytes], cfg=cfg)[0]