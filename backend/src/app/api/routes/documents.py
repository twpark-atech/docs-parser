# -*- coding: utf-8 -*-
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.pipelines.document_ingest import embedding_text, embedding_table, embedding_image
from app.parsers.documents.pdf_parser import PDFParser
from app.parsers.documents.docx_parser import DocxParser
from app.common.config import load_config
from app.common.hash import sha256_file
from app.common.parser import get_value
from app.common.runtime import now_utc
from app.storage.postgres import PostgresConfig, PostgresWriter
from app.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.storage.minio import MinIOConfig, MinIOWriter
from app.storage.embedding import EmbeddingConfig, EmbeddingProvider

_log = logging.getLogger(__name__)
router = APIRouter(tags=["document"])
JOB_TMP_DIR = Path("/tmp/docs-parser-jobs")


def _new_pg(cfg: Dict[str, Any]) -> PostgresWriter:
    dsn = str(get_value(cfg, "postgres.dsn", "")).strip()
    if not dsn:
        raise HTTPException(status_code=500, detail="postgres.dsn is not configured")
    pg = PostgresWriter(
        PostgresConfig(
            dsn=dsn,
            connect_timeout_sec=int(get_value(cfg, "postgres.connect_timeout_sec", 10)),
        )
    )
    pg.ensure_schema()
    return pg


def _get_staging_counts(cfg: Dict[str, Any], doc_id: str) -> Dict[str, int]:
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        return {"pages": 0, "images": 0, "tables": 0, "total": 0}

    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))

    index_map = {
        "pages": str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging")),
        "images": str(get_value(cfg, "opensearch.images_staging_index", "images_staging")),
        "tables": str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging")),
    }

    out: Dict[str, int] = {"pages": 0, "images": 0, "tables": 0}
    for key, index in index_map.items():
        try:
            writer = OpenSearchWriter(
                OpenSearchConfig(
                    url=os_url,
                    index=index,
                    username=os_username,
                    password=os_password,
                    verify_certs=os_verify,
                )
            )
            out[key] = int(
                writer.count_by_query(query={"query": {"term": {"doc_id": str(doc_id)}}}) or 0
            )
        except Exception as e:
            _log.warning("staging count failed: index=%s doc_id=%s err=%s", index, doc_id, e)
            out[key] = 0
    out["total"] = int(out["pages"] + out["images"] + out["tables"])
    return out


def _is_doc_fully_ingested(
    *, pg: PostgresWriter, cfg: Dict[str, Any], doc_id: str, fmt: str
) -> Tuple[bool, str]:
    errs = pg.count_ingest_errors(doc_id=doc_id)
    err_total = int(errs.get("total") or 0)
    if err_total > 0:
        return False, f"ingest errors remain: {err_total}"

    stage = _get_staging_counts(cfg, doc_id)
    if int(stage.get("total") or 0) > 0:
        return False, f"staging documents remain: {int(stage.get('total') or 0)}"

    if str(fmt).lower() == "pdf":
        progress = pg.get_doc_progress(doc_id=doc_id)
        total_pages = int((progress or {}).get("total_pages") or 0)
        contiguous_done = int((progress or {}).get("contiguous_done_until") or 0)
        if total_pages <= 0 or contiguous_done < total_pages:
            return False, f"pdf progress incomplete: {contiguous_done}/{total_pages}"

    return True, "done"


def _detect_format(filename: str, content_type: str | None) -> str:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    if name.endswith(".pdf") or ctype == "application/pdf":
        return "pdf"
    if name.endswith(".docx") or ctype in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }:
        return "docx"
    return ""


def _run_pdf_ingest(pdf_path: Path, source_name: str) -> Dict[str, Any]:
    parsed = PDFParser.parse(
        config_path=Path("config/config.yaml"),
        input_pdf=pdf_path,
        source_name=source_name,
    )
    if isinstance(parsed, dict):
        pages = parsed.get("pages")
        if isinstance(pages, list):
            meta = {k: v for k, v in parsed.items() if k != "pages"}
            return {"pdf": pages, "pdf_meta": meta}
    return {"pdf": parsed}


def _run_docx_ingest(docx_path: Path, source_name: str) -> Dict[str, Any]:
    parsed = DocxParser.parse(
        config_path=Path("config/config.yaml"),
        input_docx=docx_path,
        source_name=source_name,
    )
    if isinstance(parsed, dict):
        paragraphs = parsed.get("paragraphs")
        if isinstance(paragraphs, list):
            meta = {k: v for k, v in parsed.items() if k != "paragraphs"}
            return {"docx": paragraphs, "docx_meta": meta}
    return {"docx": parsed}


class ChunkUpdateReq(BaseModel):
    text: str = Field(min_length=1, description="Updated chunk text")


@router.get("/documents")
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    cfg = load_config(Path("config/config.yaml"))
    pg = _new_pg(cfg)
    try:
        docs = pg.list_documents(limit=limit, offset=offset)
        return {
            "status": "ok",
            "metadata": {
                "limit": int(limit),
                "offset": int(offset),
                "count": len(docs),
            },
            "documents": docs,
        }
    except Exception as e:
        _log.exception("List documents failed")
        raise HTTPException(status_code=500, detail=f"List documents failed: {e}") from e
    finally:
        pg.close()


@router.get("/documents/chunks")
async def get_document_chunks(
    doc_id: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")

    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))
    text_index = str(get_value(cfg, "opensearch.text_index", "chunks"))

    writer = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=text_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )
    try:
        res = writer.search(
            query={"term": {"doc_id": str(doc_id)}},
            size=int(limit),
            from_=int(offset),
            sort=[{"chunk_id": {"order": "asc"}}],
            source_includes=[
                "doc_id",
                "chunk_id",
                "text",
                "image_ids",
                "embedding_model",
                "ingested_at",
            ],
        )
        hits = ((res or {}).get("hits") or {}).get("hits") or []
        total_obj = ((res or {}).get("hits") or {}).get("total") or {}
        total = int(total_obj.get("value") or 0) if isinstance(total_obj, dict) else int(total_obj or 0)
        chunks = []
        for h in hits:
            src = h.get("_source") or {}
            chunks.append(
                {
                    "id": str(h.get("_id") or ""),
                    "doc_id": str(src.get("doc_id") or ""),
                    "chunk_id": str(src.get("chunk_id") or ""),
                    "text": str(src.get("text") or ""),
                    "image_ids": src.get("image_ids") or [],
                    "embedding_model": str(src.get("embedding_model") or ""),
                    "ingested_at": str(src.get("ingested_at") or ""),
                }
            )
        return {
            "status": "ok",
            "metadata": {
                "doc_id": str(doc_id),
                "index": text_index,
                "limit": int(limit),
                "offset": int(offset),
                "count": len(chunks),
                "total": total,
            },
            "chunks": chunks,
        }
    except Exception as e:
        _log.exception("Get document chunks failed")
        raise HTTPException(status_code=500, detail=f"Get document chunks failed: {e}") from e


@router.patch("/documents/chunks/{chunk_id}")
async def update_document_chunk(chunk_id: str, req: ChunkUpdateReq) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")

    text_index = str(get_value(cfg, "opensearch.text_index", "chunks"))
    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))

    writer = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=text_index,
            username=os_username,
            password=os_password,
            verify_certs=os_verify,
        )
    )

    new_text = str(req.text or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        res = writer.search(
            query={"term": {"chunk_id": str(chunk_id)}},
            size=1,
            from_=0,
            source_includes=["doc_id", "chunk_id", "text", "image_ids", "embedding_model", "ingested_at", "embedding"],
        )
        hits = ((res or {}).get("hits") or {}).get("hits") or []
        if not hits:
            raise HTTPException(status_code=404, detail="chunk not found")

        hit = hits[0] or {}
        source = dict((hit.get("_source") or {}))
        doc_id = str(source.get("doc_id") or "")
        chunk_id_src = str(source.get("chunk_id") or chunk_id)
        if not chunk_id_src:
            raise HTTPException(status_code=500, detail="chunk_id is missing in source")

        embed_base_url = str(get_value(cfg, "embed_text.base_url", "")).strip()
        embed_model = str(get_value(cfg, "embed_text.model", "")).strip()
        if not embed_base_url or not embed_model:
            raise HTTPException(status_code=500, detail="embed_text.base_url/model is not configured")

        embedder = EmbeddingProvider(
            EmbeddingConfig(
                base_url=embed_base_url,
                model=embed_model,
                timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
                max_batch_size=1,
                truncate=bool(get_value(cfg, "embed_text.truncate", True)),
            )
        )
        emb = embedder.embed([new_text])[0]

        source["doc_id"] = doc_id
        source["chunk_id"] = chunk_id_src
        source["text"] = new_text
        source["embedding_model"] = embed_model
        source["embedding"] = emb

        # 문서 적재 시점은 유지하고, 수정 시각을 별도 기록
        source["updated_at"] = now_utc()

        writer.bulk_upsert([{"_id": chunk_id_src, "_source": source}], batch_size=1)
        writer.refresh()

        return {
            "status": "ok",
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id_src,
                "index": text_index,
                "embedding_model": embed_model,
            },
            "chunk": {
                "id": chunk_id_src,
                "doc_id": doc_id,
                "chunk_id": chunk_id_src,
                "text": new_text,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("Update document chunk failed: chunk_id=%s", chunk_id)
        raise HTTPException(status_code=500, detail=f"Update document chunk failed: {e}") from e


@router.get("/documents/preview-url")
async def get_document_preview_url(
    request: Request,
    doc_id: str = Query(..., min_length=1),
    expires_sec: int = Query(default=3600, ge=1, le=86400),
    mode: str = Query(default="auto", pattern="^(auto|presigned|proxy)$"),
) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        doc = pg.get_document_by_sha256(sha256_hex=doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
    finally:
        pg.close()

    viewer_uri = str((doc or {}).get("viewer_uri") or "").strip()
    minio_key = str((doc or {}).get("minio_key") or "").strip()
    minio_bucket = str((doc or {}).get("minio_bucket") or "").strip()

    # viewer_uri가 minio://bucket/key 형식이면 key를 우선 사용
    if viewer_uri.startswith("minio://"):
        try:
            raw = viewer_uri[len("minio://") :]
            bkt, key = raw.split("/", 1)
            if bkt:
                minio_bucket = bkt
            if key:
                minio_key = key
        except Exception:
            pass

    if not minio_key:
        return {
            "status": "ok",
            "metadata": {
                "doc_id": str(doc_id),
                "preview_url": viewer_uri,
                "expires_sec": int(expires_sec),
                "kind": "viewer_uri",
            },
        }

    bucket = minio_bucket or str(get_value(cfg, "minio.bucket", "")).strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="minio.bucket is not configured")

    # MinIO presigned URL이 비-ASCII key에서 SignatureDoesNotMatch를 내는 환경이 있어
    # auto 모드에서는 proxy URL로 자동 폴백합니다.
    use_proxy = str(mode).lower() == "proxy" or (
        str(mode).lower() == "auto" and (not str(minio_key).isascii())
    )
    if use_proxy:
        preview_url = str(request.base_url).rstrip("/") + f"/api/v1/documents/preview-file?doc_id={doc_id}"
        return {
            "status": "ok",
            "metadata": {
                "doc_id": str(doc_id),
                "preview_url": preview_url,
                "expires_sec": int(expires_sec),
                "kind": "backend_proxy",
                "bucket": bucket,
                "key": minio_key,
            },
        }

    writer = MinIOWriter(
        MinIOConfig(
            endpoint=str(get_value(cfg, "minio.endpoint", "")),
            access_key=str(get_value(cfg, "minio.access_key", "")),
            secret_key=str(get_value(cfg, "minio.secret_key", "")),
            bucket=bucket,
            secure=bool(get_value(cfg, "minio.secure", False)),
        )
    )
    try:
        url = writer.presigned_get_url(object_key=minio_key, expires_sec=int(expires_sec))
        return {
            "status": "ok",
            "metadata": {
                "doc_id": str(doc_id),
                "preview_url": url,
                "expires_sec": int(expires_sec),
                "kind": "minio_presigned",
                "bucket": bucket,
                "key": minio_key,
            },
        }
    except Exception as e:
        _log.exception("Get preview url failed")
        raise HTTPException(status_code=500, detail=f"Get preview url failed: {e}") from e


@router.get("/documents/preview-file")
async def get_document_preview_file(doc_id: str = Query(..., min_length=1)) -> Response:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        doc = pg.get_document_by_sha256(sha256_hex=doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
    finally:
        pg.close()

    viewer_uri = str((doc or {}).get("viewer_uri") or "").strip()
    minio_key = str((doc or {}).get("minio_key") or "").strip()
    minio_bucket = str((doc or {}).get("minio_bucket") or "").strip()
    mime_type = str((doc or {}).get("mime_type") or "application/octet-stream").strip() or "application/octet-stream"

    if viewer_uri.startswith("minio://"):
        try:
            raw = viewer_uri[len("minio://") :]
            bkt, key = raw.split("/", 1)
            if bkt:
                minio_bucket = bkt
            if key:
                minio_key = key
        except Exception:
            pass

    if not minio_key:
        raise HTTPException(status_code=404, detail="document object key not found")

    bucket = minio_bucket or str(get_value(cfg, "minio.bucket", "")).strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="minio.bucket is not configured")

    writer = MinIOWriter(
        MinIOConfig(
            endpoint=str(get_value(cfg, "minio.endpoint", "")),
            access_key=str(get_value(cfg, "minio.access_key", "")),
            secret_key=str(get_value(cfg, "minio.secret_key", "")),
            bucket=bucket,
            secure=bool(get_value(cfg, "minio.secure", False)),
        )
    )
    try:
        payload = writer.get_object_bytes(object_key=minio_key)
        filename = minio_key.rsplit("/", 1)[-1] if "/" in minio_key else minio_key
        # HTTP header는 latin-1 제약이 있어 비-ASCII 파일명은 RFC 5987 filename*로 전달합니다.
        ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "document"
        encoded_name = quote(filename, safe="")
        headers = {
            "Content-Disposition": f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded_name}"
        }
        return Response(content=payload, media_type=mime_type, headers=headers)
    except Exception as e:
        _log.exception("Get preview file failed")
        raise HTTPException(status_code=500, detail=f"Get preview file failed: {e}") from e


def _require_confirm(confirm: bool, hint: str) -> None:
    """
    삭제 같은 파괴적 작업을 실수로 호출하는 걸 막기 위해 confirm 플래그를 강제합니다.
    - 프론트: Dialog에서 '삭제' 클릭 시에만 confirm=true로 호출
    """
    if not bool(confirm):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "confirmation required",
                "hint": hint,
                "example": hint,
            },
        )


@router.delete("/documents")
async def delete_documents_by_title(
    title: str = Query(..., min_length=1),
    confirm: bool = Query(
        default=False,
        description="Must be true to execute deletion. Prevents accidental destructive calls.",
    ),
) -> Dict[str, Any]:
    # ✅ 삭제 확인 강제
    _require_confirm(
        confirm,
        hint=f"DELETE /api/v1/documents?title={quote(title, safe='')}&confirm=true",
    )

    cfg = _cfg_with_env()

    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")
    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))
    os_indexes = [
        str(get_value(cfg, "opensearch.text_index", "chunks")),
        str(get_value(cfg, "opensearch.image_index", "images")),
        str(get_value(cfg, "opensearch.table_index", "tables")),
        str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging")),
        str(get_value(cfg, "opensearch.images_staging_index", "images_staging")),
        str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging")),
    ]

    pg = _new_pg(cfg)
    try:
        docs = pg.find_documents_by_title(title=title)
        doc_ids = [d["sha256"] for d in docs if str(d.get("sha256") or "").strip()]
        if len(doc_ids) > 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "multiple documents matched title; use DELETE /documents/{doc_id}",
                    "doc_ids": doc_ids,
                    "hint": "DELETE /api/v1/documents/{doc_id}?confirm=true",
                },
            )
        if not docs:
            return {
                "status": "ok",
                "metadata": {
                    "title": title,
                    "doc_ids": [],
                    "pg_deleted_documents": 0,
                    "minio_deleted_objects": 0,
                    "os_deleted_by_index": {},
                },
            }

        minio_writer = MinIOWriter(
            MinIOConfig(
                endpoint=str(get_value(cfg, "minio.endpoint", "")),
                access_key=str(get_value(cfg, "minio.access_key", "")),
                secret_key=str(get_value(cfg, "minio.secret_key", "")),
                bucket=str(get_value(cfg, "minio.bucket", "")),
                secure=bool(get_value(cfg, "minio.secure", False)),
            )
        )
        minio_deleted_objects = 0
        for d in docs:
            minio_key = str(d.get("minio_key") or "").strip()
            if not minio_key:
                continue
            prefix = minio_key.rsplit("/", 1)[0] if "/" in minio_key else minio_key
            try:
                minio_deleted_objects += minio_writer.delete_prefix(prefix=prefix)
            except Exception as e:
                _log.warning("MinIO delete skipped/failed: prefix=%s err=%s", prefix, e)

        os_deleted: Dict[str, int] = {idx: 0 for idx in os_indexes}
        for idx in os_indexes:
            writer = OpenSearchWriter(
                OpenSearchConfig(
                    url=os_url,
                    index=idx,
                    username=os_username,
                    password=os_password,
                    verify_certs=os_verify,
                )
            )
            for doc_id in doc_ids:
                try:
                    os_deleted[idx] += writer.delete_by_doc_id(doc_id=doc_id)
                except Exception as e:
                    _log.warning("OpenSearch delete skipped/failed: index=%s doc_id=%s err=%s", idx, doc_id, e)

        pg_deleted = pg.delete_documents_by_ids(doc_ids=doc_ids)
        return {
            "status": "ok",
            "metadata": {
                "title": title,
                "doc_ids": doc_ids,
                "pg_deleted_documents": int(pg_deleted),
                "minio_deleted_objects": int(minio_deleted_objects),
                "os_deleted_by_index": os_deleted,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("Delete documents failed")
        raise HTTPException(status_code=500, detail=f"Delete documents failed: {e}") from e
    finally:
        pg.close()


@router.delete("/documents/{doc_id}")
async def delete_document_by_id(
    doc_id: str,
    confirm: bool = Query(
        default=False,
        description="Must be true to execute deletion. Prevents accidental destructive calls.",
    ),
) -> Dict[str, Any]:
    # ✅ 삭제 확인 강제
    _require_confirm(
        confirm,
        hint=f"DELETE /api/v1/documents/{quote(doc_id, safe='')}" + "?confirm=true",
    )

    cfg = _cfg_with_env()

    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")
    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))
    os_indexes = [
        str(get_value(cfg, "opensearch.text_index", "chunks")),
        str(get_value(cfg, "opensearch.image_index", "images")),
        str(get_value(cfg, "opensearch.table_index", "tables")),
        str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging")),
        str(get_value(cfg, "opensearch.images_staging_index", "images_staging")),
        str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging")),
    ]

    pg = _new_pg(cfg)
    delete_job_id = f"del-{uuid4().hex}"
    locked = False
    try:
        pg.create_delete_job(job_id=delete_job_id, doc_id=doc_id)

        doc = pg.get_document_by_sha256(sha256_hex=doc_id)
        if doc is None:
            pg.finish_delete_job(
                job_id=delete_job_id,
                status="done",
                pg_deleted=0,
                minio_deleted=0,
                os_deleted={},
                last_error="",
            )
            return {"status": "ok", "metadata": {"doc_id": doc_id, "deleted": False}}

        minio_writer = MinIOWriter(
            MinIOConfig(
                endpoint=str(get_value(cfg, "minio.endpoint", "")),
                access_key=str(get_value(cfg, "minio.access_key", "")),
                secret_key=str(get_value(cfg, "minio.secret_key", "")),
                bucket=str(get_value(cfg, "minio.bucket", "")),
                secure=bool(get_value(cfg, "minio.secure", False)),
            )
        )
        minio_deleted_objects = 0
        minio_key = str(doc.get("minio_key") or "").strip()
        if minio_key:
            prefix = minio_key.rsplit("/", 1)[0] if "/" in minio_key else minio_key
            try:
                minio_deleted_objects = minio_writer.delete_prefix(prefix=prefix)
            except Exception as e:
                _log.warning("MinIO delete skipped/failed: prefix=%s err=%s", prefix, e)

        os_deleted: Dict[str, int] = {idx: 0 for idx in os_indexes}
        for idx in os_indexes:
            writer = OpenSearchWriter(
                OpenSearchConfig(
                    url=os_url,
                    index=idx,
                    username=os_username,
                    password=os_password,
                    verify_certs=os_verify,
                )
            )
            try:
                os_deleted[idx] += writer.delete_by_doc_id(doc_id=doc_id)
            except Exception as e:
                _log.warning("OpenSearch delete skipped/failed: index=%s doc_id=%s err=%s", idx, doc_id, e)

        pg_deleted = pg.delete_documents_by_ids(doc_ids=[doc_id])
        pg.finish_delete_job(
            job_id=delete_job_id,
            status="done",
            pg_deleted=int(pg_deleted),
            minio_deleted=int(minio_deleted_objects),
            os_deleted=os_deleted,
            last_error="",
        )
        return {
            "status": "ok",
            "metadata": {
                "delete_job_id": delete_job_id,
                "doc_id": doc_id,
                "deleted": bool(pg_deleted > 0),
                "pg_deleted_documents": int(pg_deleted),
                "minio_deleted_objects": int(minio_deleted_objects),
                "os_deleted_by_index": os_deleted,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        try:
            pg.finish_delete_job(
                job_id=delete_job_id,
                status="failed",
                pg_deleted=0,
                minio_deleted=0,
                os_deleted={},
                last_error=str(e),
            )
        except Exception:
            pass
        _log.exception("Delete document by id failed")
        raise HTTPException(status_code=500, detail=f"Delete document by id failed: {e}") from e
    finally:
        try:
            if locked and doc_id:
                pg.advisory_unlock(doc_id=doc_id)
        except Exception:
            pass
        pg.close()


@router.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str) -> Dict[str, Any]:
    cfg = _cfg_with_env()

    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")

    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))
    pages_stage_index = str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging"))
    images_stage_index = str(get_value(cfg, "opensearch.images_staging_index", "images_staging"))
    tables_stage_index = str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging"))

    pg = _new_pg(cfg)
    try:
        doc = pg.get_document_by_sha256(sha256_hex=doc_id)
        progress = pg.get_doc_progress(doc_id=doc_id)
        errs = pg.count_ingest_errors(doc_id=doc_id)
        retryable = pg.count_retryable_ingest_errors(doc_id=doc_id, max_attempts=5)

        def _count_stage(index: str) -> int:
            writer = OpenSearchWriter(
                OpenSearchConfig(
                    url=os_url,
                    index=index,
                    username=os_username,
                    password=os_password,
                    verify_certs=os_verify,
                )
            )
            return writer.count_by_query(query={"query": {"term": {"doc_id": doc_id}}})

        staging = {
            "pages": _count_stage(pages_stage_index),
            "images": _count_stage(images_stage_index),
            "tables": _count_stage(tables_stage_index),
        }
        return {
            "status": "ok",
            "metadata": {
                "doc": doc or {},
                "progress": progress or {},
                "errors": errs,
                "retryable_errors": retryable,
                "staging": staging,
            },
        }
    except Exception as e:
        _log.exception("Get document status failed")
        raise HTTPException(status_code=500, detail=f"Get document status failed: {e}") from e
    finally:
        pg.close()


@router.get("/ingest/backlog")
async def get_ingest_backlog() -> Dict[str, Any]:
    cfg = _cfg_with_env()
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")
    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", False))
    pages_stage_index = str(get_value(cfg, "opensearch.pages_staging_index", "pages_staging"))
    images_stage_index = str(get_value(cfg, "opensearch.images_staging_index", "images_staging"))
    tables_stage_index = str(get_value(cfg, "opensearch.tables_staging_index", "tables_staging"))

    pg = _new_pg(cfg)
    try:
        err_stats = pg.count_global_ingest_errors(max_attempts=5)

        def _count_stage(index: str) -> int:
            writer = OpenSearchWriter(
                OpenSearchConfig(
                    url=os_url,
                    index=index,
                    username=os_username,
                    password=os_password,
                    verify_certs=os_verify,
                )
            )
            return writer.count_by_query(query={"query": {"match_all": {}}})

        staging = {
            "pages": _count_stage(pages_stage_index),
            "images": _count_stage(images_stage_index),
            "tables": _count_stage(tables_stage_index),
        }
        return {
            "status": "ok",
            "metadata": {
                "errors": err_stats,
                "staging": staging,
            },
        }
    except Exception as e:
        _log.exception("Get ingest backlog failed")
        raise HTTPException(status_code=500, detail=f"Get ingest backlog failed: {e}") from e
    finally:
        pg.close()


def _cfg_with_env() -> Dict[str, Any]:
    cfg = load_config(Path("config/config.yaml"))
    # env 우선(민감값)
    if os.getenv("POSTGRES_DSN"):
        cfg.setdefault("postgres", {})
        cfg["postgres"]["dsn"] = os.getenv("POSTGRES_DSN")
    if os.getenv("MINIO_ACCESS_KEY"):
        cfg.setdefault("minio", {})
        cfg["minio"]["access_key"] = os.getenv("MINIO_ACCESS_KEY")
    if os.getenv("MINIO_SECRET_KEY"):
        cfg.setdefault("minio", {})
        cfg["minio"]["secret_key"] = os.getenv("MINIO_SECRET_KEY")
    if os.getenv("OS_USERNAME"):
        cfg.setdefault("opensearch", {})
        cfg["opensearch"]["username"] = os.getenv("OS_USERNAME")
    if os.getenv("OS_PASSWORD"):
        cfg.setdefault("opensearch", {})
        cfg["opensearch"]["password"] = os.getenv("OS_PASSWORD")
    if os.getenv("VLM_API_KEY"):
        cfg.setdefault("vlm", {})
        cfg["vlm"]["api_key"] = os.getenv("VLM_API_KEY")
    if os.getenv("LLM_API_KEY"):
        cfg.setdefault("llm", {})
        cfg["llm"]["api_key"] = os.getenv("LLM_API_KEY")
    return cfg


def _process_ingest_job(job_id: str) -> None:
    cfg = _cfg_with_env()
    try:
        pg = _new_pg(cfg)
    except HTTPException:
        _log.error("postgres.dsn is not configured; cannot process job=%s", job_id)
        return
    locked = False
    doc_sha = ""
    try:
        job = pg.get_ingest_job(job_id=job_id)
        if not job:
            return
        doc_sha = str(job.get("doc_id") or "")
        fmt = str(job.get("fmt") or "")
        if not doc_sha:
            pg.mark_ingest_job_failed(job_id=job_id, reason="empty doc_id")
            return

        locked = pg.advisory_try_lock(doc_id=doc_sha, timeout_sec=10)
        if not locked:
            pg.mark_ingest_job_failed(job_id=job_id, reason="advisory lock timeout")
            return
        pg.mark_ingest_job_running(job_id=job_id)

        existing_doc = pg.get_document_by_sha256(sha256_hex=doc_sha)
        if existing_doc is not None:
            done, _reason = _is_doc_fully_ingested(pg=pg, cfg=cfg, doc_id=doc_sha, fmt=fmt)
            latest_job = pg.get_latest_ingest_job_for_doc(doc_id=doc_sha)
            latest_status = str((latest_job or {}).get("status") or "")
            if done and (fmt == "pdf" or latest_status == "done"):
                pg.mark_ingest_job_done(job_id=job_id)
                return

        source_name = str(job.get("source_name") or "")
        file_path = Path(str(job.get("file_path") or ""))
        if not file_path.exists():
            pg.mark_ingest_job_failed(job_id=job_id, reason=f"file not found: {file_path}")
            return

        if fmt == "pdf":
            result = _run_pdf_ingest(file_path, source_name or file_path.name)
        elif fmt == "docx":
            result = _run_docx_ingest(file_path, source_name or file_path.name)
        else:
            pg.mark_ingest_job_failed(job_id=job_id, reason=f"unsupported fmt: {fmt}")
            return

        embedding_text(result)
        embedding_table(result)
        embedding_image(result)

        meta_key = "pdf_meta" if fmt == "pdf" else "docx_meta"
        meta = result.get(meta_key, {}) if isinstance(result, dict) else {}
        doc_id = str(meta.get("doc_id") or doc_sha)
        done, reason = _is_doc_fully_ingested(pg=pg, cfg=cfg, doc_id=doc_id, fmt=fmt)
        if done and fmt == "pdf":
            pg.clear_doc_progress(doc_id=doc_id)
        if done:
            pg.mark_ingest_job_done(job_id=job_id)
        else:
            stage = _get_staging_counts(cfg, doc_id)
            pg.mark_ingest_job_failed(
                job_id=job_id,
                reason=(
                    f"incomplete ingest: {reason}; "
                    f"staging(pages={int(stage.get('pages') or 0)},"
                    f"images={int(stage.get('images') or 0)},"
                    f"tables={int(stage.get('tables') or 0)})"
                ),
            )
    except Exception as e:
        _log.exception("Ingest job failed: job_id=%s", job_id)
        try:
            pg.mark_ingest_job_failed(job_id=job_id, reason=str(e))
        except Exception:
            pass
    finally:
        try:
            if locked and doc_sha:
                pg.advisory_unlock(doc_id=doc_sha)
        except Exception:
            pass
        pg.close()


@router.get("/ingest/jobs/{job_id}")
async def get_ingest_job(job_id: str) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        job = pg.get_ingest_job(job_id=job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"status": str(job.get("status") or "unknown"), "metadata": job}
    finally:
        pg.close()


@router.post("/document/parse")
async def document_parse(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, Any]:
    fmt = _detect_format(file.filename, file.content_type)
    if not fmt:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf or .docx")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    JOB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex
    suffix = f".{fmt}"
    tmp_path = JOB_TMP_DIR / f"{job_id}{suffix}"
    tmp_path.write_bytes(data)
    doc_sha = sha256_file(tmp_path)

    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        stale_sec = int(get_value(cfg, "ingest.stale_running_sec", 300))
        stale_marked = pg.mark_stale_running_ingest_jobs(doc_id=doc_sha, stale_after_sec=stale_sec)

        existing_doc = pg.get_document_by_sha256(sha256_hex=doc_sha)
        if existing_doc is not None:
            latest_job = pg.get_latest_ingest_job_for_doc(doc_id=doc_sha)
            latest_status = str((latest_job or {}).get("status") or "")
            is_done, _ = _is_doc_fully_ingested(pg=pg, cfg=cfg, doc_id=doc_sha, fmt=fmt)
            if is_done and latest_status in {"", "done"}:
                return {"status": "done", "metadata": existing_doc}

        pg.create_ingest_job(
            job_id=job_id,
            doc_id=doc_sha,
            source_name=file.filename or tmp_path.name,
            fmt=fmt,
            file_path=str(tmp_path),
        )
    finally:
        pg.close()

    background_tasks.add_task(_process_ingest_job, job_id)
    return {
        "status": "queued",
        "metadata": {
            "job_id": job_id,
            "doc_id": doc_sha,
            "source_name": file.filename or tmp_path.name,
            "format": fmt,
        },
    }
