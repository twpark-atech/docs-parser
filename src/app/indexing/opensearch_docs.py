# ==============================================================================
# 목적 : OpenSearch Documents 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import time
from typing import Sequence, Dict, Any, List, Tuple

from app.storage.opensearch import OpenSearchWriter


def build_os_docs(
    chunks: Sequence[Dict[str, Any]],
    vectors: Sequence[List[float]],
    *,
    embedding_model: str,    
) -> List[Dict[str, Any]]:
    """청크 메타데이터와 임베딩 벡터를 OpenSearch 문서 포맷으로 변환합니다.
    
    chunks와 vectors를 같은 순서로 zip하여 OpenSearch에 적재 가능한 문서 리스트를 생성합니다.
    ingested_at은 현재 시각을 epoch milliseconds 단위(int)로 저장합니다.

    Args:
        chunks: build_chunks_from_md 등에서 생성된 청크 dict 시퀀스.
        vectors: chunks와 1:1로 대응하는 임베딩 벡터 시퀀스.
        embedding_model: 사용한 임베딩 모델 식별자.

    Returns:
        OpenSearch bulk/index 요청에 사용할 문서 리스트.

    Raises:
        ValueError: chunks와 vectors의 길이가 다를 경우.
        KeyError: chunks 내 필수 키가 누락된 경우.
    """
    if len(chunks) != len(vectors):
        raise ValueError("normalized and vectors length mismatch")
    
    now_ms = int(time.time() * 1000)
    docs: List[Dict[str, Any]] = []

    for c, v in zip(chunks, vectors):
        docs.append(
            {
                "_id": c["chunk_id"],
                "_source": {
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "doc_title": c["doc_title"],
                    "source_uri": c["source_uri"],
                    "sha256": c["sha256"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "order": c["order"],
                    "text": c["text"],
                    "image_ids": c.get("image_ids", []),
                    "embedding": v,
                    "embedding_model": embedding_model,
                    "ingested_at": now_ms,
                },
            }
        )

    return docs


def load_pages_from_staging(
    *,
    os_pages_staging: OpenSearchWriter,
    doc_id: str,
) -> Tuple[List[str], int, int, int, Dict[str, Any]]:
    """페이지 스테이징 인덱스에서 OCR 완료 페이지를 로드하고 처리 현황을 집계합니다.
    
    doc_id가 동일하고 status="done"인 문서를 page_no 오름차순으로 조회합니다.
    각 페이지의 ocr_text를 page_texts 리스트에 담아 반환합니다.
    첫 번째 done 문서의 _source를 meta로 저장하여 함께 반환합니다.
    같은 doc_id에 대한 전체 문서를 다시 스캔하여 status를 기준으로 total/done/failed 개수를 집계합니다.

    Args:
        os_pages_staging: pages staging 인덱스에 대해 scan(query=..., size=...)을 제공하는 OpenSearchWriter.
        doc_id: 대상 문서 ID.

    Returns:
        (page_texts, total, done, failed, meta)
    """
    done_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"status": "done"}},
                ]
            }
        },
        "sort": [{"page_no": "asc"}],
    }

    page_texts: List[str] = []
    meta: Dict[str, Any] = {}
    for hit in os_pages_staging.scan(query=done_query, size=500):
        src = hit.get("_source", {}) or {}
        if not meta:
            meta = src
        page_texts.append(str(src.get("ocr_text") or ""))

    all_query = {
        "query": {"term": {"doc_id": doc_id}},
        "_source": ["status"],
    }
    total = 0
    done = 0
    failed = 0
    for hit in os_pages_staging.scan(query=all_query, size=500):
        total += 1
        st = str((hit.get("_source", {}) or {}).get("status") or "")
        if st == "done":
            done += 1
        elif st == "failed":
            failed += 1

    return page_texts, total, done, failed, meta