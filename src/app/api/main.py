from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.workflows import docx_ingest
from app.workflows import pdf_ingest

_log = logging.getLogger(__name__)

app = FastAPI(title="docs-parser")


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


def _run_pdf_ingest(pdf_path: Path) -> Dict[str, Any]:
    result = pdf_ingest.ingest_pdf(
        config_path=Path("config/config.yml"),
        input_pdf=pdf_path,
    )
    return result.to_dict()


def _run_docx_ingest(docx_path: Path) -> Dict[str, Any]:
    result = docx_ingest.ingest_docx(
        config_path=Path("config/docx_config.yml"),
        input_docx=docx_path,
    )
    return result.to_dict()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    fmt = _detect_format(file.filename, file.content_type)
    if not fmt:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf or .docx")

    suffix = f".{fmt}"
    with tempfile.TemporaryDirectory(prefix="docs-parser-") as tmpdir:
        tmp_path = Path(tmpdir) / f"upload{suffix}"
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp_path.write_bytes(data)

            if fmt == "pdf":
                return _run_pdf_ingest(tmp_path)
            return _run_docx_ingest(tmp_path)
        except HTTPException:
            raise
        except Exception as e:
            _log.exception("Ingest failed")
            raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
