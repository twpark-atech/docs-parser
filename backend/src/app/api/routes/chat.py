import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.common.config import load_config
from app.common.parser import get_value
from app.common.runtime import now_utc
from app.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.storage.postgres import PostgresConfig, PostgresWriter

_log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _cfg_with_env() -> Dict[str, Any]:
    cfg = load_config(Path("config/config.yaml"))
    if os.getenv("LLM_BASE_URL"):
        cfg.setdefault("llm", {})
        cfg["llm"]["url"] = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_API_KEY"):
        cfg.setdefault("llm", {})
        cfg["llm"]["api_key"] = os.getenv("LLM_API_KEY")
    if os.getenv("POSTGRES_DSN"):
        cfg.setdefault("postgres", {})
        cfg["postgres"]["dsn"] = os.getenv("POSTGRES_DSN")
    if os.getenv("OS_USERNAME"):
        cfg.setdefault("opensearch", {})
        cfg["opensearch"]["username"] = os.getenv("OS_USERNAME")
    if os.getenv("OS_PASSWORD"):
        cfg.setdefault("opensearch", {})
        cfg["opensearch"]["password"] = os.getenv("OS_PASSWORD")
    return cfg


def _upstream_base(cfg: Dict[str, Any]) -> str:
    return str(get_value(cfg, "llm.url", "http://211.184.184.238:8004/v1")).strip().rstrip("/")


def _upstream_api_key(cfg: Dict[str, Any]) -> str:
    return str(get_value(cfg, "llm.api_key", "")).strip()


def _mask_key(key: str) -> str:
    s = str(key or "").strip()
    if not s:
        return ""
    if len(s) <= 8:
        return s[:2] + "***"
    return s[:4] + "..." + s[-4:]


def _resolve_llm_upstream(cfg: Dict[str, Any], model: str) -> Dict[str, str]:
    """
    Resolve upstream target by requested model.
    Priority:
    1) llm.models.<name> where key == requested model OR value.model == requested model
    2) legacy llm.url / llm.api_key
    """
    req_model = str(model or "").strip()
    models = get_value(cfg, "llm.models", {}) or {}
    selected: Dict[str, Any] = {}

    if isinstance(models, dict):
        # direct key match
        if req_model and isinstance(models.get(req_model), dict):
            selected = models.get(req_model) or {}
        else:
            # fallback: search by model field
            for _, v in models.items():
                if not isinstance(v, dict):
                    continue
                if str(v.get("model") or "").strip() == req_model:
                    selected = v
                    break

    url = str(selected.get("url") or _upstream_base(cfg)).strip().rstrip("/")
    api_key = str(selected.get("api_key") or _upstream_api_key(cfg)).strip()
    return {"url": url, "api_key": api_key}


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
    _ensure_chat_schema(pg)
    return pg


def _new_chat_os(cfg: Dict[str, Any]) -> OpenSearchWriter:
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")
    writer = OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=str(get_value(cfg, "opensearch.chat_index", "chat_memory")),
            username=get_value(cfg, "opensearch.username", None),
            password=get_value(cfg, "opensearch.password", None),
            verify_certs=bool(get_value(cfg, "opensearch.verify_certs", False)),
        )
    )
    try:
        writer.ensure_index(
            body={
                "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                "mappings": {
                    "dynamic": False,
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "message_id": {"type": "keyword"},
                        "role": {"type": "keyword"},
                        "text": {"type": "text"},
                        "created_at": {"type": "date"},
                    },
                },
            }
        )
    except Exception:
        pass
    return writer


def _new_doc_os(cfg: Dict[str, Any]) -> OpenSearchWriter:
    os_url = str(get_value(cfg, "opensearch.url", "")).strip()
    if not os_url:
        raise HTTPException(status_code=500, detail="opensearch.url is not configured")
    return OpenSearchWriter(
        OpenSearchConfig(
            url=os_url,
            index=str(get_value(cfg, "opensearch.text_index", "chunks")),
            username=get_value(cfg, "opensearch.username", None),
            password=get_value(cfg, "opensearch.password", None),
            verify_certs=bool(get_value(cfg, "opensearch.verify_certs", False)),
        )
    )


def _ensure_chat_schema(pg: PostgresWriter) -> None:
    with pg.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_summaries (
                session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                summary_text TEXT NOT NULL,
                covered_until_message_id TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            """
        )
    pg.commit()


def _approx_tokens(text: str) -> int:
    s = str(text or "")
    return max(1, int((len(s) + 3) / 4)) if s else 0


def _split_chunks(text: str, chunk_chars: int = 700) -> List[str]:
    s = str(text or "").strip()
    if not s:
        return []
    out: List[str] = []
    for i in range(0, len(s), chunk_chars):
        out.append(s[i : i + chunk_chars])
    return out


def _index_message(osw: OpenSearchWriter, *, session_id: str, message_id: str, role: str, text: str, created_at: Any) -> None:
    docs = []
    chunks = _split_chunks(text)
    for i, chunk in enumerate(chunks, start=1):
        docs.append(
            {
                "_id": f"{message_id}:c{i:03d}",
                "_source": {
                    "session_id": str(session_id),
                    "message_id": str(message_id),
                    "role": str(role),
                    "text": str(chunk),
                    "created_at": created_at,
                },
            }
        )
    if docs:
        osw.bulk_upsert(docs, batch_size=200)
        osw.refresh()


async def _call_upstream_chat(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    req_model = str((payload or {}).get("model") or "").strip()
    target = _resolve_llm_upstream(cfg, req_model)
    base_url = str(target.get("url") or "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="llm.url is not configured")
    url = f"{base_url}/chat/completions"
    headers: Dict[str, str] = {}
    api_key = str(target.get("api_key") or "").strip()
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    _log.info(
        "chat upstream route: model=%s url=%s key=%s",
        req_model,
        base_url,
        _mask_key(api_key),
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(url, json=payload, headers=headers or None)
        if r.status_code >= 400:
            _log.warning("Upstream error: model=%s url=%s status=%s body=%s", req_model, base_url, r.status_code, r.text[:500])
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {r.status_code}")
        return r.json()


def _select_recent_raw(messages: List[Dict[str, Any]], keep_turns: int = 8) -> List[Dict[str, Any]]:
    return messages[-keep_turns:] if len(messages) > keep_turns else messages


async def _maybe_compress_history(
    cfg: Dict[str, Any],
    pg: PostgresWriter,
    *,
    session_id: str,
    model: str,
    threshold_tokens: int = 3000,
    keep_turns: int = 8,
) -> Dict[str, Any]:
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT message_id, role, content, token_count, created_at
            FROM chat_messages
            WHERE session_id=%s
            ORDER BY created_at ASC;
            """,
            (session_id,),
        )
        rows = cur.fetchall() or []

        cur.execute(
            """
            SELECT summary_text, covered_until_message_id
            FROM chat_summaries
            WHERE session_id=%s
            LIMIT 1;
            """,
            (session_id,),
        )
        srow = cur.fetchone()

    messages = [
        {
            "message_id": str(r[0]),
            "role": str(r[1]),
            "content": str(r[2]),
            "token_count": int(r[3] or 0),
            "created_at": str(r[4] or ""),
        }
        for r in rows
    ]
    summary_text = str((srow or [""])[0] or "")
    covered_until = str((srow or [None, ""])[1] or "")

    raw_tokens = int(sum(int(m["token_count"] or 0) for m in messages))
    if raw_tokens <= int(threshold_tokens) or len(messages) <= keep_turns:
        return {"messages": messages, "summary_text": summary_text, "covered_until": covered_until}

    old = messages[:-keep_turns]
    recent = messages[-keep_turns:]
    old_text = "\n".join([f"[{m['role']}] {m['content']}" for m in old])

    summarize_prompt = (
        "아래는 동일 세션의 과거 대화 기록이다. 사실/선호/결정사항/할 일/중요 맥락을 유지하며 간결하게 한국어 요약을 작성하라. "
        "추정 금지, 중복 제거, 핵심만 bullet 형태로 정리하라.\n\n"
        f"[기존 요약]\n{summary_text}\n\n"
        f"[새로 압축할 대화]\n{old_text}"
    )

    resp = await _call_upstream_chat(
        cfg,
        {
            "model": model,
            "messages": [{"role": "user", "content": summarize_prompt}],
            "temperature": 0.0,
            "max_tokens": 700,
            "stream": False,
        },
    )

    new_summary = _extract_assistant_text(resp)

    if new_summary:
        now = now_utc()
        covered_until = str(old[-1]["message_id"]) if old else covered_until
        with pg.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_summaries(session_id, summary_text, covered_until_message_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    summary_text=EXCLUDED.summary_text,
                    covered_until_message_id=EXCLUDED.covered_until_message_id,
                    updated_at=EXCLUDED.updated_at;
                """,
                (session_id, new_summary, covered_until, now, now),
            )
        pg.commit()
        summary_text = new_summary

    return {"messages": recent, "summary_text": summary_text, "covered_until": covered_until}


def _rag_search(osw: OpenSearchWriter, *, session_id: str, query_text: str, top_k: int = 6) -> List[Dict[str, Any]]:
    q = {
        "bool": {
            "must": [{"term": {"session_id": str(session_id)}}],
            "should": [{"match": {"text": {"query": str(query_text), "operator": "or"}}}],
            "minimum_should_match": 1,
        }
    }
    res = osw.search(
        query=q,
        size=max(1, int(top_k)),
        from_=0,
        sort=[{"_score": {"order": "desc"}}, {"created_at": {"order": "desc"}}],
        source_includes=["message_id", "role", "text", "created_at"],
    )
    hits = ((res or {}).get("hits") or {}).get("hits") or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            {
                "id": str(h.get("_id") or ""),
                "score": float(h.get("_score") or 0.0),
                "message_id": str(src.get("message_id") or ""),
                "role": str(src.get("role") or ""),
                "text": str(src.get("text") or ""),
                "created_at": str(src.get("created_at") or ""),
            }
        )
    return out


def _rag_search_docs(
    osw: OpenSearchWriter,
    *,
    query_text: str,
    doc_ids: List[str],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    cleaned_ids = [str(x).strip() for x in (doc_ids or []) if str(x).strip()]
    if not cleaned_ids:
        return []
    q: Dict[str, Any] = {
        "bool": {
            "must": [
                {"terms": {"doc_id": cleaned_ids}},
                {"match": {"text": {"query": str(query_text), "operator": "or"}}},
            ]
        }
    }
    res = osw.search(
        query=q,
        size=max(1, int(top_k)),
        from_=0,
        sort=[{"_score": {"order": "desc"}}, {"chunk_id": {"order": "asc"}}],
        source_includes=["doc_id", "chunk_id", "text"],
    )
    hits = ((res or {}).get("hits") or {}).get("hits") or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            {
                "id": str(h.get("_id") or ""),
                "score": float(h.get("_score") or 0.0),
                "doc_id": str(src.get("doc_id") or ""),
                "chunk_id": str(src.get("chunk_id") or ""),
                "text": str(src.get("text") or ""),
            }
        )
    return out


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts: List[str] = []
        for item in v:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
                    continue
                t = item.get("content")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(v, dict):
        t = v.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()
        c = v.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
    return str(v).strip()


def _extract_text_deep(v: Any, depth: int = 0) -> str:
    if depth > 8 or v is None:
        return ""
    if isinstance(v, str):
        s = v.strip()
        return s
    if isinstance(v, list):
        parts: List[str] = []
        for item in v:
            t = _extract_text_deep(item, depth + 1)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    if isinstance(v, dict):
        # Never expose internal reasoning traces to end users.
        if "reasoning_content" in v:
            v = {k: vv for k, vv in v.items() if k != "reasoning_content"}

        for key in ("content", "text", "output_text"):
            if key in v:
                t = _extract_text_deep(v.get(key), depth + 1)
                if t:
                    return t

        # OpenAI responses-like shape: output[].content[].text
        if "output" in v:
            t = _extract_text_deep(v.get("output"), depth + 1)
            if t:
                return t

        # chat completions-like shape: choices[0].message / delta / text
        if "choices" in v:
            t = _extract_text_deep(v.get("choices"), depth + 1)
            if t:
                return t

        for kk, vv in v.items():
            k = str(kk).lower()
            if "reasoning" in k or "think" in k or "analysis" in k:
                continue
            if ("text" in k) or ("content" in k) or ("message" in k) or ("answer" in k):
                t = _extract_text_deep(vv, depth + 1)
                if t:
                    return t
    return ""


def _extract_assistant_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if isinstance(choices, list) and choices:
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = c0.get("message") if isinstance(c0, dict) else None
        if isinstance(msg, dict):
            # Guard: do not leak internal reasoning traces.
            if msg.get("content") is None and msg.get("reasoning_content"):
                return ""
            text = _to_text(msg.get("content"))
            if text:
                return text
        delta = c0.get("delta") if isinstance(c0, dict) else None
        if isinstance(delta, dict):
            text = _to_text(delta.get("content"))
            if text:
                return text
        text = _to_text(c0.get("text") if isinstance(c0, dict) else "")
        if text:
            return text

    text = _to_text(resp.get("message")) if isinstance(resp, dict) else ""
    if text:
        return text
    text = _to_text(resp.get("content")) if isinstance(resp, dict) else ""
    if text:
        return text
    text = _to_text(resp.get("output_text")) if isinstance(resp, dict) else ""
    if text:
        return text
    return _extract_text_deep(resp)


def _extract_finish_reason(resp: Dict[str, Any]) -> str:
    try:
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0] if isinstance(choices[0], dict) else {}
            fr = c0.get("finish_reason")
            if isinstance(fr, str):
                return fr
    except Exception:
        pass
    return ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionReq(BaseModel):
    model: str = Field(default="gpt-oss-20b")
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    doc_ids: Optional[list[str]] = None


class SessionCreateReq(BaseModel):
    title: Optional[str] = None


class SessionMessageReq(BaseModel):
    content: str
    model: str = Field(default="gpt-oss-20b")
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    doc_ids: Optional[list[str]] = None


class SessionSearchReq(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=20)


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionReq) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    payload = req.model_dump()
    payload.pop("doc_ids", None)
    try:
        return await _call_upstream_chat(cfg, payload)
    except httpx.RequestError as e:
        _log.exception("Upstream request failed")
        raise HTTPException(status_code=502, detail=f"Upstream LLM request failed: {e}") from e


@router.post("/chat/sessions")
async def create_session(req: SessionCreateReq) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        sid = uuid4().hex
        now = now_utc()
        title = str(req.title or "새 대화").strip() or "새 대화"
        with pg.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions(session_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s);
                """,
                (sid, title, now, now),
            )
        pg.commit()
        return {"status": "ok", "metadata": {"session_id": sid, "title": title}}
    finally:
        pg.close()


@router.get("/chat/sessions")
async def list_sessions(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    m.role AS last_message_role,
                    m.content AS last_message,
                    m.created_at AS last_message_at
                FROM chat_sessions s
                LEFT JOIN LATERAL (
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = s.session_id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) m ON TRUE
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s OFFSET %s;
                """,
                (int(limit), int(offset)),
            )
            rows = cur.fetchall() or []
        sessions = [
            {
                "session_id": str(r[0] or ""),
                "title": str(r[1] or ""),
                "created_at": str(r[2] or ""),
                "updated_at": str(r[3] or ""),
                "last_message_role": str(r[4] or ""),
                "last_message": str(r[5] or ""),
                "last_message_at": str(r[6] or ""),
            }
            for r in rows
        ]
        return {"status": "ok", "metadata": {"count": len(sessions), "limit": int(limit), "offset": int(offset)}, "sessions": sessions}
    finally:
        pg.close()


@router.get("/chat/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    try:
        with pg.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE session_id=%s LIMIT 1;", (session_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="session not found")
            cur.execute(
                """
                SELECT message_id, role, content, token_count, created_at
                FROM chat_messages
                WHERE session_id=%s
                ORDER BY created_at ASC
                LIMIT %s OFFSET %s;
                """,
                (session_id, int(limit), int(offset)),
            )
            rows = cur.fetchall() or []
        messages = [
            {
                "message_id": str(r[0] or ""),
                "role": str(r[1] or ""),
                "content": str(r[2] or ""),
                "token_count": int(r[3] or 0),
                "created_at": str(r[4] or ""),
            }
            for r in rows
        ]
        return {
            "status": "ok",
            "metadata": {"session_id": session_id, "count": len(messages), "limit": int(limit), "offset": int(offset)},
            "messages": messages,
        }
    finally:
        pg.close()


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    osw = _new_chat_os(cfg)
    deleted = False
    deleted_os = 0
    try:
        with pg.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE session_id=%s LIMIT 1;", (session_id,))
            exists = bool(cur.fetchone())
            if not exists:
                raise HTTPException(status_code=404, detail="session not found")

            # chat_messages / chat_summaries are deleted via FK cascade
            cur.execute("DELETE FROM chat_sessions WHERE session_id=%s;", (session_id,))
            deleted = (cur.rowcount or 0) > 0
        pg.commit()

        # OpenSearch 메모리 인덱스 정리(실패해도 DB 삭제 결과는 유지)
        try:
            res = osw._client.delete_by_query(  # type: ignore[attr-defined]
                index=osw.index,
                body={"query": {"term": {"session_id": str(session_id)}}},
                refresh=True,
                conflicts="proceed",
            )
            deleted_os = int((res or {}).get("deleted") or 0)
        except Exception as e:
            _log.warning("chat memory delete_by_query failed: session_id=%s err=%s", session_id, e)

        return {
            "status": "ok",
            "metadata": {
                "session_id": session_id,
                "deleted": bool(deleted),
                "deleted_chat_memory": int(deleted_os),
            },
        }
    finally:
        pg.close()


@router.post("/chat/sessions/{session_id}/search")
async def search_session_memory(session_id: str, req: SessionSearchReq) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    osw = _new_chat_os(cfg)
    try:
        hits = _rag_search(osw, session_id=session_id, query_text=req.query, top_k=req.top_k)
        return {"status": "ok", "metadata": {"session_id": session_id, "count": len(hits)}, "hits": hits}
    except Exception as e:
        _log.exception("session memory search failed")
        raise HTTPException(status_code=500, detail=f"session memory search failed: {e}") from e


@router.post("/chat/sessions/{session_id}/messages")
async def send_session_message(session_id: str, req: SessionMessageReq) -> Dict[str, Any]:
    cfg = _cfg_with_env()
    pg = _new_pg(cfg)
    osw = _new_chat_os(cfg)
    doc_osw = _new_doc_os(cfg)

    try:
        with pg.cursor() as cur:
            cur.execute("SELECT title FROM chat_sessions WHERE session_id=%s LIMIT 1;", (session_id,))
            srow = cur.fetchone()
            if not srow:
                raise HTTPException(status_code=404, detail="session not found")

        now = now_utc()
        user_message_id = uuid4().hex
        user_text = str(req.content or "").strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="content is empty")

        with pg.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages(message_id, session_id, role, content, token_count, created_at)
                VALUES (%s, %s, 'user', %s, %s, %s);
                """,
                (user_message_id, session_id, user_text, _approx_tokens(user_text), now),
            )
            cur.execute("UPDATE chat_sessions SET updated_at=%s WHERE session_id=%s;", (now, session_id))
        pg.commit()

        _index_message(osw, session_id=session_id, message_id=user_message_id, role="user", text=user_text, created_at=now)

        compressed = await _maybe_compress_history(
            cfg,
            pg,
            session_id=session_id,
            model=str(req.model),
            threshold_tokens=3000,
            keep_turns=8,
        )

        recent = compressed.get("messages") or []
        summary_text = str(compressed.get("summary_text") or "")

        rag_hits = _rag_search(osw, session_id=session_id, query_text=user_text, top_k=6)
        rag_context = "\n".join([f"- ({h['role']}) {h['text']}" for h in rag_hits])
        selected_doc_ids = [str(x).strip() for x in (req.doc_ids or []) if str(x).strip()]
        doc_hits: List[Dict[str, Any]] = []
        if selected_doc_ids:
            try:
                doc_hits = _rag_search_docs(doc_osw, query_text=user_text, doc_ids=selected_doc_ids, top_k=8)
            except Exception as e:
                _log.warning("doc rag search failed: doc_ids=%s err=%s", selected_doc_ids, e)
        doc_context = "\n".join(
            [f"- [doc:{h['doc_id']}] [chunk:{h['chunk_id']}] {h['text']}" for h in doc_hits]
        )

        system_rules = (
            "너는 세션형 어시스턴트다. 이전 대화를 기억해서 일관되게 답하라. "
            "근거 없는 추정은 금지하고, 과거 대화 질문은 제공된 검색 근거를 우선 사용하라."
        )
        memory_block = (
            f"[장기 요약 메모리]\n{summary_text or '(없음)'}\n\n"
            f"[세션 RAG 검색 근거]\n{rag_context or '(없음)'}\n\n"
            f"[지식 저장소 RAG 검색 근거]\n{doc_context or '(없음)'}"
        )

        messages_for_llm: List[Dict[str, str]] = [
            {"role": "system", "content": system_rules},
            {"role": "system", "content": memory_block},
        ]
        for m in recent:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if role in {"user", "assistant", "system"} and content:
                messages_for_llm.append({"role": role, "content": content})

        payload = {
            "model": str(req.model),
            "messages": messages_for_llm,
            "temperature": float(req.temperature if req.temperature is not None else 0.2),
            "max_tokens": int(req.max_tokens if req.max_tokens is not None else 1024),
            "stream": bool(req.stream) if req.stream is not None else False,
        }
        llm_resp = await _call_upstream_chat(cfg, payload)

        answer = _extract_assistant_text(llm_resp)
        finish_reason = _extract_finish_reason(llm_resp)

        # 일부 upstream은 내부 추론 토큰에서 먼저 길이 제한에 걸려 content가 비는 경우가 있어 1회 재시도.
        if not answer and finish_reason == "length":
            retry_payload = dict(payload)
            retry_payload["max_tokens"] = min(max(int(payload["max_tokens"]) * 2, 2048), 4096)
            retry_payload["messages"] = messages_for_llm + [
                {"role": "system", "content": "최종 답변만 한국어로 간결하게 출력하라. 내부 추론 과정은 출력하지 마라."}
            ]
            llm_resp = await _call_upstream_chat(cfg, retry_payload)
            answer = _extract_assistant_text(llm_resp)
            finish_reason = _extract_finish_reason(llm_resp)

        if not answer:
            _log.warning(
                "upstream empty assistant text: finish_reason=%s keys=%s body=%s",
                finish_reason,
                list((llm_resp or {}).keys()),
                str(llm_resp)[:800],
            )
            raise HTTPException(
                status_code=502,
                detail=f"empty assistant response from upstream (finish_reason={finish_reason or 'unknown'})",
            )

        assistant_message_id = uuid4().hex
        now2 = now_utc()
        with pg.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages(message_id, session_id, role, content, token_count, created_at)
                VALUES (%s, %s, 'assistant', %s, %s, %s);
                """,
                (assistant_message_id, session_id, answer, _approx_tokens(answer), now2),
            )
            cur.execute("UPDATE chat_sessions SET updated_at=%s WHERE session_id=%s;", (now2, session_id))
        pg.commit()

        _index_message(
            osw,
            session_id=session_id,
            message_id=assistant_message_id,
            role="assistant",
            text=answer,
            created_at=now2,
        )

        return {
            "status": "ok",
            "metadata": {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "rag_hits": len(rag_hits),
                "doc_rag_hits": len(doc_hits),
                "doc_ids": selected_doc_ids,
                "used_summary": bool(summary_text),
            },
            "assistant": {
                "role": "assistant",
                "content": answer,
            },
        }
    finally:
        pg.close()
