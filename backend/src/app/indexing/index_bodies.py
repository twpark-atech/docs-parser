# ==============================================================================
# 목적 : Index Body 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-16
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations
from typing import Dict, Any

from app.parsing.regex import HNSW_4096, HNSW_1024


def build_chunks_body() -> Dict[str, Any]:
    """PDF 텍스트 청크 인덱스(pdf_chunks v1) 생성용 OpenSearch body를 반환합니다.
    
    문서에서 추출한 텍스트 청크를 저장/검색하기 위한 인덱스 스키마입니다.
    KNN 벡터 검색을 위해 index.knn=True를 활성화하며, embedding 필드를 HNSW 벡터 매핑으로 정의합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {
            "index": {
            "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":           { "type": "keyword" },
            "chunk_id":         { "type": "keyword" },
            "text":             { "type": "text" },
            "image_ids":        { "type": "keyword" },
            "embedding":        HNSW_4096,
            "embedding_model":  { "type": "keyword" },
            "ingested_at":      { "type": "date" }
            }
        }
    }


def build_pages_staging_body() -> Dict[str, Any]:
    """PDF 페이지 OCR 결과 스테이징 인덱스(pdf_chunks_staging v1) 생성용 OpenSearch body를 반환합니다.
    
    페이지 단위 OCR 처리 상태를 관리하기 위한 스테이징 인덱스입니다.
    벡터 검색이 필요 없으므로 index.knn=False이며, 처리 속도/비용을 위해 단일 샤드(1)와 레플리카(0)를 사용합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {"index": {"knn": False, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":        { "type": "keyword" },
            "page_id":       { "type": "keyword" },
            "doc_title":     { "type": "text" },
            "source_uri":    { "type": "keyword" },
            "pdf_uri":       { "type": "keyword" },
            "page_no":       { "type": "integer" },
            "ocr_text":      { "type": "text" },
            "ocr_model":     { "type": "keyword" },
            "prompt":        { "type": "text" },
            "prompt_sha256": { "type": "keyword" },
            "status":        { "type": "keyword" },
            "attempts":      { "type": "integer" },
            "last_error":    { "type": "text" },
            "created_at":    { "type": "date" },
            "updated_at":    { "type": "date" }
            }
        }
    }


def build_images_body() -> Dict[str, Any]:
    """PDF 이미지 인덱스(pdf_images v1) 생성용 OpenSearch body를 반환합니다.
    
    문서에서 추출된 이미지의 메타데이터와 이미지/설명 임베딩을 저장/검색하기 위한 인덱스 스키마입니다.
    KNN 벡터 검색을 위해 index.knn=True를 활성화하며, embedding 필드를 HNSW 벡터 매핑으로 정의합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":       { "type": "keyword" },
            "image_id":     { "type": "keyword" },
            "desc_text":             { "type": "text" },
            "desc_embedding":        HNSW_4096,
            "image_embedding":       HNSW_1024,
            "desc_embedding_model":  { "type": "keyword" },
            "image_embedding_model": { "type": "keyword" },
            "ingested_at":           { "type": "date" }
            }
        }
    }


def build_images_staging_body() -> Dict[str, Any]:
    """PDF 이미지 스테이징 인덱스(pdf_images_staging v1) 생성용 OpenSearch body를 반환합니다.
    
    이미지 추출 및 설명 생성 전/중/후의 처리 상태를 관리하는 스테이징 인덱스입니다.
    벡터 검색이 필요 없으므로 index.knn=False이며, 처리 속도/비용을 위해 단일 샤드(1)와 레플리카(0)를 사용합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {"index": {"knn": False, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":       { "type": "keyword" },
            "image_id":     { "type": "keyword" },
            "doc_title":    { "type": "text" },
            "source_uri":   { "type": "keyword" },
            "page_no":      { "type": "integer" },
            "order":        { "type": "integer" },
            "pdf_uri":      { "type": "keyword" },
            "image_uri":    { "type": "keyword" },
            "image_mime":   { "type": "keyword" },
            "image_sha256": { "type": "keyword" },
            "width":        { "type": "integer" },
            "height":       { "type": "integer" },
            "bbox": {
                "properties": {
                "x1": { "type": "integer" },
                "y1": { "type": "integer" },
                "x2": { "type": "integer" },
                "y2": { "type": "integer" }
                }
            },
            "desc_text":    { "type": "text" },
            "status":       {"type": "keyword"},
            "attempts":     {"type": "integer"},
            "last_error":   {"type": "text"},
            "created_at":   {"type": "date"},
            "updated_at":   { "type": "date" }
            }
        }
    }


def build_tables_body() -> Dict[str, Any]:
    """PDF 테이블 인덱스(pdf_tables v1) 생성용 OpenSearch body를 반환합니다.
    
    문서에서 내 테이블 데이터를 저장하고 행 단위 벡터 검색을 위한 인덱스 스키마입니다.
    KNN 벡터 검색을 위해 index.knn=True를 활성화하며, embedding 필드를 HNSW 벡터 매핑으로 정의합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":       { "type": "keyword" },
            "table_id":     { "type": "keyword" },
            "table_text":   { "type": "text" },
            "row_embedding": HNSW_4096,
            "row_embedding_model":  { "type": "keyword" },
            "ingested_at":          { "type": "date" }
            }
        }
    }


def build_tables_staging_body() -> Dict[str, Any]:
    """PDF 테이블 스테이징 인덱스(pdf_tables_staging v1) 생성용 OpenSearch body를 반환합니다.
    
    테이블 추출 전후의 중간 산출물과 처리 상태를 관리하는 스테이징 인덱스입니다.
    벡터 검색이 필요 없으므로 index.knn=False이며, 처리 속도/비용을 위해 단일 샤드(1)와 레플리카(0)를 사용합니다.
    dynamic=False로 정의되지 않은 필드는 저장되지 않도록 스키마를 고정합니다.

    Returns:
        OpenSearch index 생성 요청에 사용할 dict({"settings": ..., "mappings": ...}).
    """
    return {
        "settings": {"index": {"knn": False, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic": False,
            "properties": {
            "doc_id":       { "type": "keyword" },
            "table_id":     { "type": "keyword" },

            "doc_title":    { "type": "text" },
            "source_uri":   { "type": "keyword" },
            "pdf_uri":      { "type": "keyword" },

            "page_no":      { "type": "integer" },
            "order":        { "type": "integer" },
            
            "table_sha256": { "type": "keyword" },
            "raw_html":     { "type": "text" },

            "header_json":  { "type": "text" },
            "rows_json":    { "type": "text" },

            "status":       { "type": "keyword" },
            "attempts":     { "type": "integer" },
            "last_error":   { "type": "text" },
            "created_at":   { "type": "date" },
            "updated_at":   { "type": "date" }
            }
        }
    }