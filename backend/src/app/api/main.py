import logging

from fastapi import FastAPI

from app.api.routes.documents import router as docs_router
from app.api.routes.chat import router as chat_router
from fastapi.middleware.cors import CORSMiddleware

_log = logging.getLogger(__name__)

app = FastAPI(
    title="Document Parser",
    description="Document parse and retrieval system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple health check endpoint for container health monitoring"""
    return {"status": "healthy", "service": "docs-parser"}

app.include_router(docs_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
