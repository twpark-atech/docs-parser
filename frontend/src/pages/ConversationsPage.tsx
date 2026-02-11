import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  IconButton,
  MenuItem,
  Popover,
  Select,
  Typography,
  TextField,
  InputBase,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import AddIcon from "@mui/icons-material/Add";
import Inventory2OutlinedIcon from "@mui/icons-material/Inventory2Outlined";
import BuildOutlinedIcon from "@mui/icons-material/BuildOutlined";

import CalendarMonthOutlinedIcon from "@mui/icons-material/CalendarMonthOutlined";
import ColorLensOutlinedIcon from "@mui/icons-material/ColorLensOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import EditNoteOutlinedIcon from "@mui/icons-material/EditNoteOutlined";
import ReactMarkdown from "react-markdown";
import { useSearchParams } from "react-router-dom";

import { useQuery } from "@tanstack/react-query";
import { fetchDocuments } from "../api/documents";
import { ApiError, createChatSession, fetchSessionMessages, sendSessionMessage } from "../api/chat";
import type { ChatMessage } from "../api/chat";

const LS_MODEL_KEY = "chat.selectedModel";
const MODEL_OPTIONS = [
  { id: "gpt-oss-20b", label: "gpt-oss-20b" },
  { id: "Qwen3-VL-32B-Instruct-AWQ", label: "Qwen3-VL-32B-Instruct-AWQ" }
];

type DocRow = {
  sha256: string;
  title: string;
  mime_type?: string;
  size_bytes?: number;
};

export default function ConversationsPage() {
  const defaultModel = useMemo(() => MODEL_OPTIONS[0]?.id ?? "gpt-oss-20b", []);
  const [model, setModel] = useState(defaultModel);

  // ===== messages (진짜 채팅) =====
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionBooting, setSessionBooting] = useState(false);
  const [searchParams] = useSearchParams();
  const requestedSessionId = (searchParams.get("session_id") || "").trim();

  // ===== auto scroll =====
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // ===== 지식 저장소 Popover =====
  const [kbAnchorEl, setKbAnchorEl] = useState<HTMLElement | null>(null);
  const kbOpen = Boolean(kbAnchorEl);
  const openKb = (e: React.MouseEvent<HTMLElement>) => setKbAnchorEl(e.currentTarget);
  const closeKb = () => setKbAnchorEl(null);

  // ===== 문서 선택 state =====
  const [docSearch, setDocSearch] = useState("");
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  const docsQ = useQuery({
    queryKey: ["documents", 200, 0],
    queryFn: () => fetchDocuments(200, 0),
    enabled: kbOpen,
    staleTime: 10_000,
  });

  const allDocs: DocRow[] = useMemo(() => (docsQ.data?.documents ?? []) as DocRow[], [docsQ.data]);

  const filteredDocs = useMemo(() => {
    const q = docSearch.trim().toLowerCase();
    if (!q) return allDocs;
    return allDocs.filter((d) => (d.title ?? "").toLowerCase().includes(q));
  }, [allDocs, docSearch]);

  const allFilteredSelected = useMemo(() => {
    if (!filteredDocs.length) return false;
    return filteredDocs.every((d) => selectedDocIds.includes(d.sha256));
  }, [filteredDocs, selectedDocIds]);

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) => (prev.includes(docId) ? prev.filter((x) => x !== docId) : [...prev, docId]));
  };

  const toggleSelectAllFiltered = () => {
    setSelectedDocIds((prev) => {
      const filteredIds = filteredDocs.map((d) => d.sha256);
      const isAll = filteredIds.length > 0 && filteredIds.every((id) => prev.includes(id));
      if (isAll) return prev.filter((id) => !filteredIds.includes(id));
      const set = new Set(prev);
      filteredIds.forEach((id) => set.add(id));
      return Array.from(set);
    });
  };

  useEffect(() => {
    const saved = localStorage.getItem(LS_MODEL_KEY);
    if (saved && MODEL_OPTIONS.some((m) => m.id === saved)) setModel(saved);
  }, []);

  useEffect(() => {
    let mounted = true;

    const boot = async () => {
      setSessionBooting(true);
      try {
        if (!requestedSessionId) {
          if (!mounted) return;
          setSessionId("");
          setMessages([]);
          return;
        }
        const loaded = await fetchSessionMessages(requestedSessionId, 500, 0);
        if (!mounted) return;
        const serverMessages: ChatMessage[] = (loaded?.messages ?? []).map((m) => ({
          role: m.role,
          content: m.content,
        }));
        setSessionId(requestedSessionId);
        setMessages(serverMessages);
      } catch (err) {
        if (!mounted) return;
        if (err instanceof ApiError && err.status === 404) {
          setSessionId("");
          setMessages([]);
        } else {
          setMessages([{ role: "assistant", content: "❌ 세션 로드 실패" }]);
        }
      } finally {
        if (mounted) setSessionBooting(false);
      }
    };
    void boot();
    return () => {
      mounted = false;
    };
  }, [requestedSessionId]);

  const onChangeModel = (e: any) => {
    const next = e.target.value as string;
    setModel(next);
    localStorage.setItem(LS_MODEL_KEY, next);
  };

  // ✅ 새 메시지 들어오면 자동 스크롤 (채팅화면일 때만)
  useEffect(() => {
    if (messages.length === 0) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending]);

  const onSend = async () => {
    const text = message.trim();
    if (!text || sending || sessionBooting) return;

    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setMessage("");
    setSending(true);

    try {
      let sid = sessionId;
      if (!sid) {
        setSessionBooting(true);
        const created = await createChatSession("새 대화");
        sid = String(created?.metadata?.session_id || "").trim();
        if (!sid) throw new Error("session create failed");
        setSessionId(sid);
      }

      const send = (sid: string) =>
        sendSessionMessage({
          session_id: sid,
          content: text,
          model,
          temperature: 0.2,
          max_tokens: 1024,
          stream: false,
          doc_ids: selectedDocIds,
        });

      const resp = await send(sid);

      const assistant = resp?.assistant?.content || "(empty)";
      setMessages((prev) => [...prev, { role: "assistant", content: assistant }]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ 오류: ${err?.message ?? String(err)}` },
      ]);
    } finally {
      setSessionBooting(false);
      setSending(false);
    }
  };

  const examples = [
    { text: "이번주 일정 알려줘", icon: <CalendarMonthOutlinedIcon /> },
    { text: "염색 레시피 추천해줘", icon: <ColorLensOutlinedIcon /> },
    { text: "베어링 결합 종류에 대해서 알려줘", icon: <SettingsOutlinedIcon /> },
    { text: "회의록 내용 정리해줘", icon: <EditNoteOutlinedIcon /> },
  ];

  const empty = messages.length === 0;

  // ✅ 채팅 시작 전에는 가운데 카드(640), 채팅 시작 후에는 body 폭에 맞춤(스크린샷 요구)
  const containerMaxW = empty ? 640 : 920;

  return (
    <Box sx={{ height: "100%", minHeight: 560, px: 2, py: 4 }}>
      <Box sx={{ width: "100%", maxWidth: 920, mx: "auto" }}>
        {/* Header (empty 상태에서만 중앙 타이틀) */}
        {empty && (
          <>
            <Box sx={{ textAlign: "center", mb: 2.5 }}>
              <Typography sx={{ fontSize: 30, fontWeight: 900, color: "#111827" }}>
                대화
              </Typography>
              <Typography sx={{ mt: 0.8, color: "#6b7280" }}>
                문서를 선택하고 질문해보세요.
              </Typography>
            </Box>

            <Box sx={{ display: "flex", justifyContent: "center", gap: 1.2, mb: 2.5 }}>
              <Chip
                icon={<Inventory2OutlinedIcon />}
                label={`지식 저장소${selectedDocIds.length ? ` (${selectedDocIds.length})` : ""}`}
                variant="outlined"
                onClick={openKb}
                sx={{
                  borderRadius: 999,
                  px: 0.5,
                  bgcolor: "#fff",
                  cursor: "pointer",
                  "&:hover": { bgcolor: "#f9fafb" },
                }}
              />
              <Chip
                icon={<BuildOutlinedIcon />}
                label="도구"
                variant="outlined"
                sx={{ borderRadius: 999, px: 0.5, bgcolor: "#fff" }}
              />
            </Box>
          </>
        )}

        {/* messages 영역 */}
        {!empty && (
          <Box
            ref={scrollRef}
            sx={{
              mx: "auto",
              width: "100%",
              maxWidth: containerMaxW,
              border: "1px solid #eef0f3",
              borderRadius: 3,
              bgcolor: "#fff",
              p: 2,
              mb: 2,
              height: 520,
              overflowY: "auto",
            }}
          >
            {messages.map((m, idx) => (
              <Box
                key={idx}
                sx={{
                  mb: 1.2,
                  display: "flex",
                  justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <Box
                  sx={{
                    maxWidth: "78%",
                    px: 1.5,
                    py: 1.1,
                    borderRadius: 2.5,
                    bgcolor: m.role === "user" ? "#2563eb" : "#f3f4f6",
                    color: m.role === "user" ? "#fff" : "#111827",
                    lineHeight: 1.5,
                    fontSize: 14,
                  }}
                >
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <Typography component="p" sx={{ m: 0, fontSize: 14, lineHeight: 1.5 }}>{children}</Typography>,
                      ul: ({ children }) => <Box component="ul" sx={{ my: 0.5, pl: 2.5 }}>{children}</Box>,
                      ol: ({ children }) => <Box component="ol" sx={{ my: 0.5, pl: 2.5 }}>{children}</Box>,
                      li: ({ children }) => <Box component="li" sx={{ my: 0.25 }}>{children}</Box>,
                      code: ({ children }) => (
                        <Box
                          component="code"
                          sx={{
                            px: 0.6,
                            py: 0.2,
                            borderRadius: 1,
                            bgcolor: m.role === "user" ? "rgba(255,255,255,0.18)" : "#e5e7eb",
                            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                            fontSize: 12,
                          }}
                        >
                          {children}
                        </Box>
                      ),
                      pre: ({ children }) => (
                        <Box
                          component="pre"
                          sx={{
                            m: 0,
                            mt: 0.8,
                            p: 1,
                            borderRadius: 1.5,
                            overflowX: "auto",
                            bgcolor: m.role === "user" ? "rgba(255,255,255,0.12)" : "#e5e7eb",
                          }}
                        >
                          {children}
                        </Box>
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                </Box>
              </Box>
            ))}

            {sending && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
                <CircularProgress size={16} />
                <Typography sx={{ fontSize: 12, color: "#6b7280" }}>응답 생성 중…</Typography>
              </Box>
            )}

            {/* ✅ 스크롤 목표점 */}
            <div ref={bottomRef} />
          </Box>
        )}

        {/* ✅ 지식 저장소 Popover (문서만) */}
        <Popover
          open={kbOpen}
          anchorEl={kbAnchorEl}
          onClose={closeKb}
          anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
          transformOrigin={{ vertical: "top", horizontal: "center" }}
          PaperProps={{
            sx: {
              mt: 1,
              borderRadius: 2.5,
              border: "1px solid #eef0f3",
              boxShadow: "0 12px 28px rgba(0,0,0,0.08)",
              p: 2,
              minWidth: 420,
              bgcolor: "#fff",
            },
          }}
        >
          <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 1.25 }}>
            <Box>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.8 }}>
                <Typography sx={{ fontSize: 12, fontWeight: 900, color: "#111827" }}>
                  문서
                </Typography>
                <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
                  선택됨 {selectedDocIds.length}
                </Typography>
              </Box>

              <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
                <TextField
                  size="small"
                  placeholder="문서 검색"
                  value={docSearch}
                  onChange={(e) => setDocSearch(e.target.value)}
                  sx={{ flex: 1 }}
                  InputProps={{ sx: { borderRadius: 2, fontSize: 13 } }}
                />

                <Button
                  variant="outlined"
                  onClick={toggleSelectAllFiltered}
                  sx={{
                    borderRadius: 2,
                    height: 40,
                    minWidth: 120,
                    borderColor: "#dbeafe",
                    color: "#2563eb",
                    fontWeight: 900,
                    bgcolor: "#fff",
                    "&:hover": { bgcolor: "#eff6ff", borderColor: "#bfdbfe" },
                  }}
                  disabled={docsQ.isLoading || docsQ.isError}
                >
                  {allFilteredSelected ? "전체 해제" : "전체 선택"}
                </Button>
              </Box>

              <Box
                sx={{
                  border: "1px solid #eef0f3",
                  borderRadius: 2,
                  overflow: "hidden",
                  maxHeight: 260,
                  overflowY: "auto",
                }}
              >
                {docsQ.isLoading && (
                  <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
                    <CircularProgress size={16} />
                    <Typography sx={{ fontSize: 12, color: "#6b7280" }}>문서 불러오는 중…</Typography>
                  </Box>
                )}

                {docsQ.isError && (
                  <Box sx={{ p: 2 }}>
                    <Typography sx={{ fontSize: 12, color: "#ef4444", fontWeight: 800 }}>
                      문서 로드 실패
                    </Typography>
                    <Typography sx={{ fontSize: 12, color: "#6b7280", mt: 0.5 }}>
                      백엔드(/api/v1/documents) 또는 프록시/CORS 확인
                    </Typography>
                  </Box>
                )}

                {!docsQ.isLoading && !docsQ.isError && !filteredDocs.length && (
                  <Box sx={{ p: 2 }}>
                    <Typography sx={{ fontSize: 12, color: "#6b7280" }}>
                      문서가 없습니다. (문서 관리에서 업로드하세요)
                    </Typography>
                  </Box>
                )}

                {!docsQ.isLoading &&
                  !docsQ.isError &&
                  filteredDocs.map((d) => {
                    const checked = selectedDocIds.includes(d.sha256);
                    return (
                      <Box
                        key={d.sha256}
                        onClick={() => toggleDoc(d.sha256)}
                        sx={{
                          display: "grid",
                          gridTemplateColumns: "36px 1fr",
                          alignItems: "center",
                          px: 1,
                          py: 0.9,
                          borderBottom: "1px solid #f1f5f9",
                          cursor: "pointer",
                          bgcolor: checked ? "#eef2ff" : "#fff",
                          "&:hover": { bgcolor: checked ? "#eef2ff" : "#f9fafb" },
                        }}
                      >
                        <Checkbox
                          checked={checked}
                          onChange={() => toggleDoc(d.sha256)}
                          onClick={(e) => e.stopPropagation()}
                          size="small"
                        />
                        <Box sx={{ minWidth: 0 }}>
                          <Typography
                            sx={{
                              fontSize: 13,
                              fontWeight: 800,
                              color: "#111827",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {d.title}
                          </Typography>
                          <Typography sx={{ fontSize: 11, color: "#6b7280", mt: 0.2 }}>
                            {d.mime_type ?? "unknown"}
                            {typeof d.size_bytes === "number" ? ` · ${(d.size_bytes / 1024).toFixed(1)} KB` : ""}
                          </Typography>
                        </Box>
                      </Box>
                    );
                  })}
              </Box>

              <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 1 }}>
                <Button size="small" variant="contained" onClick={closeKb} sx={{ borderRadius: 2, fontWeight: 900 }}>
                  적용
                </Button>
              </Box>
            </Box>
          </Box>
        </Popover>

        {/* ✅ Input bar (채팅 시작 전/후 너비 동기화) */}
        <Box
          sx={{
            mx: "auto",
            width: "100%",
            maxWidth: containerMaxW, // ✅ body와 동일 폭
            bgcolor: "#fff",
            border: "1px solid #e5e7eb",
            borderRadius: 2.5,
            px: 1.5,
            py: 1.25,
            boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
          }}
        >
          {/* top: textarea */}
          <InputBase
            multiline
            minRows={2}
            maxRows={6}
            placeholder="질문을 입력해주세요."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
            sx={{
              width: "100%",
              fontSize: 14,
              lineHeight: 1.5,
              color: "#111827",
              "& textarea": {
                width: "100%",
                resize: "none",
                padding: "10px 6px 6px 6px",
                boxSizing: "border-box",
                background: "transparent",
              },
              "& textarea::placeholder": {
                color: "#9ca3af",
                opacity: 1,
              },
            }}
          />

          <Box sx={{ height: 1, bgcolor: "#eef0f3", my: 1 }} />

          {/* bottom: controls */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <IconButton
              size="small"
              onClick={() => {
                if (sending || sessionBooting) return;
                // 새 대화 시작: 세션은 즉시 만들지 않고, 첫 전송 시 생성한다.
                setSessionId("");
                setMessages([]);
                setMessage("");
              }}
              sx={{
                border: "1px solid #e5e7eb",
                borderRadius: 2,
                width: 34,
                height: 34,
              }}
            >
              <AddIcon fontSize="small" />
            </IconButton>

            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              {empty && (
                <>
                  <Typography sx={{ fontSize: 11, color: "#6b7280", whiteSpace: "nowrap" }}>
                    LLM 모델
                  </Typography>

                  <Select
                    size="small"
                    value={model}
                    onChange={onChangeModel}
                    sx={{
                      height: 34,
                      minWidth: 140,
                      borderRadius: 2,
                      "& .MuiOutlinedInput-notchedOutline": { borderColor: "#e5e7eb" },
                      "& .MuiSelect-select": {
                        display: "flex",
                        alignItems: "center",
                        height: 34,
                        py: 0,
                        fontSize: 13,
                        pr: 3.5,
                      },
                    }}
                  >
                    {MODEL_OPTIONS.map((m) => (
                      <MenuItem key={m.id} value={m.id}>
                        {m.label}
                      </MenuItem>
                    ))}
                  </Select>
                </>
              )}

              <IconButton
                size="small"
                onClick={onSend}
                disabled={sending || sessionBooting}
                sx={{ width: 34, height: 34, color: "#2563eb" }}
              >
                <SendIcon fontSize="small" />
              </IconButton>
            </Box>
          </Box>
        </Box>

        {/* examples (empty 상태에서만) */}
        {empty && (
          <Box sx={{ textAlign: "center", mt: 4 }}>
            <Typography sx={{ color: "#6b7280", fontWeight: 800, fontSize: 12, mb: 2 }}>
              예시 질문
            </Typography>

            <Box
              sx={{
                mx: "auto",
                width: "100%",
                maxWidth: 760,
                display: "grid",
                gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                gap: 2,
              }}
            >
              {examples.map((ex) => (
                <Button
                  key={ex.text}
                  variant="outlined"
                  onClick={() => setMessage(ex.text)}
                  startIcon={ex.icon}
                  sx={{
                    justifyContent: "flex-start",
                    borderRadius: 3,
                    py: 2,
                    px: 2,
                    bgcolor: "#fff",
                    borderColor: "#e5e7eb",
                    color: "#111827",
                    "&:hover": { bgcolor: "#f9fafb", borderColor: "#d1d5db" },
                  }}
                >
                  <Typography sx={{ fontWeight: 700, fontSize: 14, textAlign: "left" }}>
                    {ex.text}
                  </Typography>
                </Button>
              ))}
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
}
