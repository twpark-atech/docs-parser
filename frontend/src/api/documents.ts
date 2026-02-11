// src/api/documents.ts
import { apiFetch } from "./client";

const V1 = "/api/v1";

export type ListDocumentsResponse = {
  status: string;
  metadata: { limit: number; offset: number; count: number };
  documents: any[];
};

export async function fetchDocuments(limit = 50, offset = 0): Promise<ListDocumentsResponse> {
  return apiFetch(`${V1}/documents?limit=${limit}&offset=${offset}`);
}

export type ChunksResponse = {
  status: string;
  metadata: {
    doc_id: string;
    index: string;
    limit: number;
    offset: number;
    count: number;
    total: number;
  };
  chunks: Array<{
    id: string;
    doc_id: string;
    chunk_id: string;
    text: string;
    image_ids: string[];
    embedding_model: string;
    ingested_at: string;
  }>;
};

export async function fetchChunksByDocId(docId: string, limit = 50, offset = 0): Promise<ChunksResponse> {
  return apiFetch(`${V1}/documents/chunks?doc_id=${encodeURIComponent(docId)}&limit=${limit}&offset=${offset}`);
}

export async function updateChunkText(chunkId: string, text: string) {
  return apiFetch(`${V1}/documents/chunks/${encodeURIComponent(chunkId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export type PreviewUrlResponse = {
  status: string;
  metadata: any;
  preview_url?: string; // 백엔드가 metadata.preview_url로 내려주는 경우도 있어서 아래에서 처리
};

export async function fetchPreviewUrlByDocId(docId: string, expiresSec = 3600, mode: "auto" | "presigned" | "proxy" = "auto") {
  const res: any = await apiFetch(
    `${V1}/documents/preview-url?doc_id=${encodeURIComponent(docId)}&expires_sec=${expiresSec}&mode=${mode}`
  );
  // 호환: {metadata:{preview_url}} 형태
  const preview_url = res?.preview_url ?? res?.metadata?.preview_url;
  return { ...res, preview_url };
}

export async function deleteDocumentByTitle(title: string) {
  return apiFetch(`${V1}/documents?title=${encodeURIComponent(title)}`, { method: "DELETE" });
}

export async function deleteDocumentById(docId: string) {
  return apiFetch(`${V1}/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
}

/**
 * ✅ ParsePage.tsx가 찾는 export
 * FastAPI: POST /api/v1/document/parse  (multipart/form-data, field name = "file")
 */
export async function parseDocument(file: File) {
  const fd = new FormData();
  fd.append("file", file);

  return apiFetch(`${V1}/document/parse`, {
    method: "POST",
    body: fd,
    // Content-Type은 fetch가 boundary 포함해서 자동 지정하므로 넣지마
  });
}
