import { useMemo, useState } from "react";
import { Box, Button, Checkbox, CircularProgress, Divider, IconButton, TextField, Typography } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { deleteChatSession, listChatSessions } from "../api/chat";

export default function ChatListPage() {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [deleting, setDeleting] = useState(false);

  const sessionsQ = useQuery({
    queryKey: ["chat-sessions", 200, 0],
    queryFn: () => listChatSessions(200, 0),
    staleTime: 10_000,
  });

  const filtered = useMemo(() => {
    const all = sessionsQ.data?.sessions ?? [];
    const needle = q.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((s) => String(s.title || "").toLowerCase().includes(needle));
  }, [sessionsQ.data, q]);

  const toggleSelect = (sessionId: string) => {
    setSelectedIds((prev) => (prev.includes(sessionId) ? prev.filter((id) => id !== sessionId) : [...prev, sessionId]));
  };

  const handleDeleteOne = async (sessionId: string) => {
    if (!window.confirm("이 채팅을 삭제할까요?")) return;
    setDeleting(true);
    try {
      await deleteChatSession(sessionId);
      setSelectedIds((prev) => prev.filter((id) => id !== sessionId));
      await sessionsQ.refetch();
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (!selectedIds.length) return;
    if (!window.confirm(`선택한 ${selectedIds.length}개 채팅을 삭제할까요?`)) return;
    setDeleting(true);
    try {
      await Promise.all(selectedIds.map((id) => deleteChatSession(id)));
      setSelectedIds([]);
      await sessionsQ.refetch();
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Typography sx={{ fontSize: 22, fontWeight: 800 }}>내 채팅 기록</Typography>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Button
            variant={selectionMode ? "outlined" : "contained"}
            sx={{ borderRadius: 2 }}
            onClick={() => {
              setSelectionMode((v) => !v);
              setSelectedIds([]);
            }}
          >
            {selectionMode ? "선택 취소" : "선택"}
          </Button>
          {selectionMode && (
            <Button
              variant="contained"
              color="error"
              sx={{ borderRadius: 2 }}
              disabled={!selectedIds.length || deleting}
              onClick={handleDeleteSelected}
            >
              선택 삭제
            </Button>
          )}
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            sx={{ borderRadius: 2 }}
            onClick={() => navigate("/conversations")}
          >
            새 채팅
          </Button>
        </Box>
      </Box>

      <TextField
        fullWidth
        size="small"
        placeholder="세션 제목 검색"
        sx={{ mb: 2 }}
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />

      <Box sx={{ border: "1px solid #eef0f3", borderRadius: 2, overflow: "hidden", bgcolor: "#fff" }}>
        {sessionsQ.isLoading && (
          <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
            <CircularProgress size={16} />
            <Typography sx={{ fontSize: 12, color: "#6b7280" }}>채팅 기록 불러오는 중…</Typography>
          </Box>
        )}

        {sessionsQ.isError && (
          <Box sx={{ p: 2 }}>
            <Typography sx={{ fontSize: 12, color: "#ef4444", fontWeight: 800 }}>채팅 기록 로드 실패</Typography>
          </Box>
        )}

        {!sessionsQ.isLoading && !sessionsQ.isError && filtered.length === 0 && (
          <Typography sx={{ color: "#6b7280", py: 4, textAlign: "center" }}>채팅 기록이 없습니다.</Typography>
        )}

        {!sessionsQ.isLoading &&
          !sessionsQ.isError &&
          filtered.map((s) => (
            <Box key={s.session_id}>
              <Box
                sx={{
                  px: 2,
                  py: 1.3,
                  display: "grid",
                  gridTemplateColumns: selectionMode ? "32px 1fr 36px" : "1fr 36px",
                  alignItems: "center",
                  gap: 0.5,
                  cursor: "pointer",
                  "&:hover": { bgcolor: "#f9fafb" },
                }}
                onClick={() => {
                  if (selectionMode) toggleSelect(s.session_id);
                  else navigate(`/conversations?session_id=${encodeURIComponent(s.session_id)}`);
                }}
              >
                {selectionMode && (
                  <Checkbox
                    size="small"
                    checked={selectedIds.includes(s.session_id)}
                    onChange={() => toggleSelect(s.session_id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                )}
                <Box sx={{ minWidth: 0 }}>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: "#111827" }}>{s.title || "새 대화"}</Typography>
                  <Typography
                    sx={{
                      fontSize: 12,
                      color: "#374151",
                      mt: 0.4,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {(s.last_message || "").trim() || "(대화 내용 없음)"}
                  </Typography>
                  <Typography sx={{ fontSize: 11, color: "#6b7280", mt: 0.35 }}>
                    {s.updated_at ? `updated: ${s.updated_at}` : ""}
                  </Typography>
                </Box>
                <IconButton
                  size="small"
                  color="error"
                  disabled={deleting}
                  onClick={(e) => {
                    e.stopPropagation();
                    void handleDeleteOne(s.session_id);
                  }}
                >
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              </Box>
              <Divider />
            </Box>
          ))}
      </Box>
    </Box>
  );
}
