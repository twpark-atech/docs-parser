// src/api/chat.ts

export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type SessionMeta = {
  session_id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  last_message_role?: ChatRole | "";
  last_message?: string;
  last_message_at?: string;
};

export type SessionMessage = {
  message_id: string;
  role: ChatRole;
  content: string;
  token_count: number;
  created_at: string;
};

export type CreateChatCompletionRequest = {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  doc_ids?: string[];
};

export type CreateChatCompletionResponse = {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices?: Array<{
    index?: number;
    message?: ChatMessage;
    delta?: Partial<ChatMessage>;
    finish_reason?: string | null;
  }>;
  usage?: any;

  // optional fallback shapes
  message?: ChatMessage;
  content?: string;
};

const V1 = "/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorText(res: Response) {
  try {
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await res.json();
      return JSON.stringify(j);
    }
    return await res.text();
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function createChatCompletion(
  req: CreateChatCompletionRequest
): Promise<CreateChatCompletionResponse> {
  const res = await fetch(`${V1}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: req.model,
      messages: req.messages,
      temperature: req.temperature ?? 0.2,
      max_tokens: req.max_tokens ?? 1024,
      stream: req.stream ?? false,
      doc_ids: req.doc_ids ?? [],
    }),
  });

  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }

  return (await res.json()) as CreateChatCompletionResponse;
}

export function extractAssistantText(resp: CreateChatCompletionResponse): string {
  const t1 = resp?.choices?.[0]?.message?.content;
  if (typeof t1 === "string" && t1.trim()) return t1;

  const t2 = resp?.message?.content;
  if (typeof t2 === "string" && t2.trim()) return t2;

  const t3 = (resp as any)?.content;
  if (typeof t3 === "string" && t3.trim()) return t3;

  return "";
}

export async function createChatSession(title?: string): Promise<{ status: string; metadata: SessionMeta }> {
  const res = await fetch(`${V1}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title ?? "새 대화" }),
  });
  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; metadata: SessionMeta };
}

export async function listChatSessions(
  limit = 50,
  offset = 0
): Promise<{ status: string; metadata: any; sessions: SessionMeta[] }> {
  const res = await fetch(`${V1}/chat/sessions?limit=${limit}&offset=${offset}`);
  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; metadata: any; sessions: SessionMeta[] };
}

export async function fetchSessionMessages(
  sessionId: string,
  limit = 200,
  offset = 0
): Promise<{ status: string; metadata: any; messages: SessionMessage[] }> {
  const res = await fetch(
    `${V1}/chat/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}&offset=${offset}`
  );
  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; metadata: any; messages: SessionMessage[] };
}

export async function sendSessionMessage(req: {
  session_id: string;
  content: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  doc_ids?: string[];
}): Promise<{
  status: string;
  metadata: any;
  assistant: ChatMessage;
}> {
  const res = await fetch(`${V1}/chat/sessions/${encodeURIComponent(req.session_id)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content: req.content,
      model: req.model,
      temperature: req.temperature ?? 0.2,
      max_tokens: req.max_tokens ?? 1024,
      stream: req.stream ?? false,
      doc_ids: req.doc_ids ?? [],
    }),
  });
  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; metadata: any; assistant: ChatMessage };
}

export async function deleteChatSession(sessionId: string): Promise<{ status: string; metadata: any }> {
  const res = await fetch(`${V1}/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const msg = await readErrorText(res);
    throw new ApiError(res.status, msg || `Request failed: ${res.status}`);
  }
  return (await res.json()) as { status: string; metadata: any };
}
