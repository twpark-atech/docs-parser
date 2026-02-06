# ==============================================================================
# 목적 : 통합 Index Schema 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-26
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict

from app.parsing.regex import HNSW_1024, HNSW_4096


def _settings() -> Dict[str, Any]:
    return {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    }


def build_chunks_body_backup() -> Dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "chunk_id":        { "type": "keyword" },
                "order":           { "type": "integer" },
                "page_start":      { "type": "integer" },
                "page_end":        { "type": "integer" },
                "text":            { "type": "text" },
                "image_ids":       { "type": "keyword" },
                "embedding_model": { "type": "keyword" },
                "embedding":       HNSW_4096,
                "ingested_at":     { "type": "date" },
            },
        },
    }


def build_chunks_body() -> Dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "chunk_id":        { "type": "keyword" },
                "text":            { "type": "text" },
                "embedding_model": { "type": "keyword" },
                "embedding":       HNSW_4096,
                "ingested_at":     { "type": "date" },
            },
        },
    }


def build_images_body() -> Dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "image_id":        { "type": "keyword" },
                "stage_id":        { "type": "keyword" },
                "page_no":         { "type": "integer" },
                "order":           { "type": "integer" },
                "image_uri":       { "type": "keyword" },
                "image_mime":      { "type": "keyword" },
                "image_sha256":    { "type": "keyword" },
                "width":           { "type": "integer" },
                "height":          { "type": "integer" },
                "bbox":            { "type": "object", "enabled": True },
                "desc_text":       { "type": "text" },
                "desc_embedding_model":  { "type": "keyword" },
                "image_embedding_model": { "type": "keyword" },
                "image_embedding": HNSW_1024,
                "desc_embedding":  HNSW_4096,
                "ingested_at":     { "type": "date" },
            },
        },
    }


def build_tables_body() -> Dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_type":        { "type": "keyword" },
                "doc_id":          { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "page_no":         { "type": "integer" },
                "order":           { "type": "integer" },
                "table_id":        { "type": "keyword" },
                "table_sha256":    { "type": "keyword" },
                "raw_html":        { "type": "text" },
                "header":          { "type": "keyword" },
                "row_count":       { "type": "integer" },
                "col_count":       { "type": "integer" },
                "table_text":      { "type": "text" },
                "row_idx":         { "type": "integer" },
                "row_obj_json":    { "type": "text" },
                "row_text":        { "type": "text" },
                "row_embedding_model": { "type": "keyword" },
                "row_embedding":   HNSW_4096,
                "ingested_at":     { "type": "date" },
            },
        },
    }


def build_pages_staging_body() -> Dict[str, Any]:
    return {
        "settings": _settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "page_id":         { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "page_no":         { "type": "integer" },
                "page_text":       { "type": "text" },
                "ocr_model":       { "type": "keyword" },
                "prompt":          { "type": "text" },
                "status":          { "type": "keyword" },
                "attempts":        { "type": "integer" },
                "last_error":      { "type": "text" },
                "created_at":      { "type": "date" },
                "updated_at":      { "type": "date" },
            },
        },
    }


def build_images_staging_body() -> Dict[str, Any]:
    return {
        "settings": _settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "stage_id":        { "type": "keyword" },
                "image_id":        { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "page_no":         { "type": "integer" },
                "order":           { "type": "integer" },
                "image_uri":       { "type": "keyword" },
                "image_mime":      { "type": "keyword" },
                "image_sha256":    { "type": "keyword" },
                "width":           { "type": "integer" },
                "height":          { "type": "integer" },
                "bbox":            { "type": "object", "enabled": True },
                "desc_text":       { "type": "text" },
                "status":          { "type": "keyword" },
                "attempts":        { "type": "integer" },
                "last_error":      { "type": "text" },
                "created_at":      { "type": "date" },
                "updated_at":      { "type": "date" },
            },
        },
    }


def build_tables_staging_body() -> Dict[str, Any]:
    return {
        "settings": _settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "doc_id":          { "type": "keyword" },
                "table_id":        { "type": "keyword" },
                "doc_title":       { "type": "keyword" },
                "source_uri":      { "type": "keyword" },
                "viewer_uri":      { "type": "keyword" },
                "source_type":     { "type": "keyword" },
                "page_no":         { "type": "integer" },
                "order":           { "type": "integer" },
                "table_sha256":    { "type": "keyword" },
                "raw_html":        { "type": "text" },
                "header_json":     { "type": "text" },
                "rows_json":       { "type": "text" },
                "status":          { "type": "keyword" },
                "attempts":        { "type": "integer" },
                "last_error":      { "type": "text" },
                "created_at":      { "type": "date" },
                "updated_at":      { "type": "date" },
            },
        },
    }