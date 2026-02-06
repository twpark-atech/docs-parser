# ==============================================================================
# 목적 : Embedding 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import requests
from dataclasses import dataclass
from typing import List, Optional

from app.ports.embedding import EmbeddingProvider
from app.common.runtime import request_with_retry


@dataclass(frozen=True)
class OllamaEmbeddingConfig:
    """Ollama EmbeddingProvider 설정 값 클래스.
    Ollama /api/embed 호출에 필요한 기본 정보와 운영 파라미터를 담습니다.
    
    Attributes:
        base_url: Ollama 서버 기본 URL.
        model: 사용할 임베딩 모델명.
        timeout_sec: 요청 타임아웃(초).
        max_batch_size: 최대 배치 크기.
        truncate: 입력이 길 때 잘라서 처리할지 여부.
    """
    base_url: str
    model: str
    timeout_sec: int = 120
    max_batch_size: int = 16
    truncate: bool = True
    api_key: Optional[str] = None


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama /api/embed 기반 텍스트 임베딩 제공자 클래스.
    
    EmbeddingProvider 계약(입력 순서 보존, 1:1 대응)을 만족하도록 구현되어 있습니다.
    embed(): texts를 max_batch_size 단위로 분할 호출하여 벡터를 순서대로 이어 붙여 반환합니다.
    _embed_batch(): 단일 배치를 /api/embed로 요쳥하고, requests_with_retry로 실패 시 재시도합니다.
    """
    def __init__(self, cfg: OllamaEmbeddingConfig):
        """설정(cfg)으로 Provider를 초기화합니다.
        
        Args:
            cfg: OllamaEmbeddingConfig 설정 객체.
        """
        self._cfg = cfg
        self._endpoint = cfg.base_url.rstrip("/") + "/api/embed"
        self._openai_endpoint = self._build_openai_endpoint(cfg.base_url)

    @staticmethod
    def _build_openai_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/v1/embeddings"):
            return base
        if base.endswith("/v1"):
            return base + "/embeddings"
        return base + "/v1/embeddings"

    @staticmethod
    def _auth_headers(api_key: Optional[str]) -> dict:
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩하여 벡터 리스트로 반환합니다.

        입력 texts를 cfg.max_batch_size 단위로 분할해 _embed_batch를 호출합니다.
        각 배치의 결과를 순서대로 합쳐 최종 벡터 리스트를 반환합니다.

        Args:
            texts: 임베딩할 텍스트 리스트.

        Returns:
            입력 texts와 동일한 순서/개수로 대응되는 임베딩 벡터 리스트.

        Raises:
            requests.exception.RequestException: 네트워크/요청 레벨에서 오류가 발생한 경우.
            requests.HTTPError: 4xx/5xx 응답이 발생한 경우.
            ValueError: Ollama가 빈 embeddings를 반환하거나 응답 형식이 잘못된 경우.
        """
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._cfg.max_batch_size):
            batch = texts[i : i + self._cfg.max_batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors
    
    def _embed_batch(self, inputs: List[str]) -> List[List[float]]:
        """단일 배치 입력을 Ollama /api/embed로 호출해 embeddings를 반환합니다.
        
        request_with_retry를 사용해 호출 실패 시 재시도합니다.
        
        Args:
            inputs: 단일 배치 텍스트 리스트.

        Returns:
            inputs와 1:1로 대응되는 임베딩 벡터 리스트.

        Raises:
            ValueError: embeddings가 비어있거나 형식이 잘못된 경우, embeddings 개수가 inputs와 다를 경우.
            RuntimeError: request_with_retry가 최종 실패할 경우.
            requests.exception.RequestException: 네트워크/요청 레벨에서 오류가 발생한 경우.
            requests.HTTPError: 4xx/5xx 응답이 발생한 경우.
        """
        def _call():
            if "/v1" in self._cfg.base_url:
                payload = {
                    "model": self._cfg.model,
                    "input": inputs,
                }
                r = requests.post(
                    self._openai_endpoint,
                    json=payload,
                    headers=self._auth_headers(self._cfg.api_key),
                    timeout=self._cfg.timeout_sec,
                )
                r.raise_for_status()
                data = r.json()
                data_items = data.get("data") or []
                if not isinstance(data_items, list) or not data_items:
                    raise ValueError(f"OpenAI compat returned empty embeddings: {data}")
                return [d.get("embedding") for d in data_items]

            payload = {
                "model": self._cfg.model,
                "input": inputs,
                "truncate": self._cfg.truncate,
            }
            r = requests.post(self._endpoint, json=payload, timeout=self._cfg.timeout_sec)
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if not isinstance(embs, list) or not embs:
                raise ValueError(f"Ollama returned empty embeddings: {data}")
            return embs
        
        embs = request_with_retry(_call, retries=5)
        if len(embs) != len(inputs):
            raise ValueError(f"Embeddings count mismatch: {len(embs)} != {len(inputs)}")
        return embs
