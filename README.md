# docs-parser

FastAPI 기반 문서 인게스트 서비스.

## 실행
```
PYTHONPATH=src uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# 또는 Docker Compose
docker compose up -d --build
```

## 사용
```
# PDF
curl -F "file=@/path/to/doc.pdf" http://localhost:8000/ingest

# DOCX
curl -F "file=@/path/to/doc.docx" http://localhost:8000/ingest
```

## 설정
- PDF: `config/config.yml`
- DOCX: `config/docx_config.yml`
