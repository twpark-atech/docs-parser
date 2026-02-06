# ==============================================================================
# 목적 : Qdrant 유틸 (현재 사용 X, 추후 Chunk Size 커지면 사용)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import logging, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant 접속/컬렉션 설정 클래스.
    QdrantClient 초기화에 필요한 URL/API key와 벡터 컬렉션 생성 시 사용할 distance metric을 보관합니다.

    Attributes:
        url: Qdrant 서버 URL.
        collection: 사용할 컬렉션명.
        distance: 거리 메트릭 문자열("COSINE" | "EUCLID" | "DOT").
        api_key: Qdrant API Key.
    """
    url: str
    collection: str
    distance: str = "COSINE"
    api_key: Optional[str] = None


class QdrantWriter:
    """Qdrant 컬렉션 생성 및 포인트 업서트를 수행하는 Writer 클래스.

    ensure_collection(vector_size): 컬렉션이 없으면 생성하고, 있으면 그대로 사용합니다.
    distance metric(COSINE/EUCLID/DOT)과 vector dimension(size)을 VectorParams로 설정합니다.
    upsert(points): points를 batch_size 단위로 분할하여 upsert 합니다.
    _normalize_point_id(raw_id): Qdrant가 허용하는 id 타입을 정규화하고, 안정적인 업서트를 위해 결정적 UUID5로 변환할 수 있습니다.

    Raises:
        ValueError: distance가 허용되지 않는 값이거나, vector 누락/ID가 비정상인 경우.
    """
    _UUID_NAMESPACE = uuid.UUID("3d6f0a2e-3b22-4b4e-8ef0-5f9a7c1c2b11")

    def __init__(self, cfg: QdrantConfig):
        """Writer를 초기화하고 QdrantClient를 생성합니다.
        
        Args:
            cfg: Qdrantconfig 설정 객체.
        """
        self._cfg = cfg
        self._client = QdrantClient(url=cfg.url, api_key=cfg.api_key)

    def ensure_collection(self, vector_size: int) -> None:
        """컬렉션이 존재하도록 보장합니다.
        
        cfg.distance를 검증하여 COSINE/EUCLID/DOT만 허용합니다.
        get_collection이 성공하면 이미 존재하므로 종료합니다.
        실패하면 create_collection로 vectors_config(VectorParams)를 구성하여 생성합니다.

        Args:
            vector_size: 벡터 차원(size).

        Returns:
            None

        Raises:
            ValueError: cfg.distance 값이 허용 목록에 없을 경우.
        """
        dist_key = (self._cfg.distance or "").strip().upper()
        allowed = {"COSINE", "EUCLID", "DOT"}
        if dist_key not in allowed:
            raise ValueError(f"Invalid distance='{self._cfg.distnace}'. Allowed: {sorted(allowed)}")
        
        try:
            self._client.get_collection(self._cfg.collection)
            return
        except Exception as e:
            _log.info("get_collection failed (will try create): %s", e)

        dist = getattr(models.Distance, self._cfg.distance)
        self._client.create_collection(
            collection_name=self._cfg.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=dist),
        )

    def upsert(self, points: List[Dict[str, Any]], batch_size: int = 256, wait: bool = True) -> None:
        """포인트 리스트를 Qdrant에 배치 업서트합니다.
        
        Args:
            points: 업서트할 포인트 dict 리스트.
            batch_size: 한 번에 업서트할 포인트 수.
            wait: True면 서버 반영 완료까지 대기.
        
        Returns:
            None

        Raises:
            ValueError: vector 누락, id가 음수 int, 빈 문자열 id 등 입력 검증에 실패할 경우.
        """
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            qd_points = [self._to_point_struct(p) for p in chunk]
            self._client.upsert(
                collection_name=self._cfg.collection,
                points=qd_points,
                wait=wait,
            )
            _log.info("Qdrant upserted: %d", len(chunk))

    def _to_point_struct(self, p: Dict[str, Any]):
        """입력 dict를 Qdrant PointStruct로 변환합니다.
        
        Args:
            p: {"id": ..., "vector": ..., "payload": ...} 형태의 dict.

        Returns:
            models.PointStructure(id=..., vector=..., payload=...)

        Raises:
            ValueError: vector가 없을 때.
        """
        raw_id = p.get("id")
        vector = p.get("vector")
        payload = dict(p.get("payload", {}) or {})

        if vector is None:
            raise ValueError("Point is missing required field: 'vector'")
        
        qdrant_id, source_id = self._normalize_point_id(raw_id)
        
        if source_id is not None:
            payload.setdefault("source_id", source_id)

        return models.PointStruct(
            id=qdrant_id,
            vector=vector,
            payload=payload,
        )
    
    def _normalize_point_id(self, raw_id: Any) -> Tuple[Union[int, str], Optional[str]]:
        """Qdrant 포인트 id를 허용 타입으로 정규화합니다.
        
        Args:
            raw_id: 원본 id 값.

        Returns:
            (qdrant_id, source_id)

        Raises:
            ValueError: 음수 int, 빈 문자열 id, str 변환 후 빈 값 등의 경우.
        """
        if isinstance(raw_id, int):
            if raw_id < 0:
                raise ValueError(f"Invalid point id (negative int): {raw_id}")
            return raw_id, None
        
        if isinstance(raw_id, uuid.UUID):
            return str(raw_id), None
        
        if isinstance(raw_id, str):
            s = raw_id.strip()
            if not s:
                raise ValueError("Point id is empty string. Provide stable id per chunk for upsert.")
            
            try:
                return str(uuid.UUID(s)), None
            except Exception:
                deterministic = uuid.uuid5(self._UUID_NAMESPACE, s)
                return str(deterministic), s
            
        s = str(raw_id)
        if not s:
            raise ValueError(f"Point id is invalid type and empty after str(): {type(raw_id)}")
        deterministic = uuid.uuid5(self._UUID_NAMESPACE, s)
        return str(deterministic), s