import {
  Box,
  Button,
  Divider,
  IconButton,
  InputAdornment,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import RefreshIcon from "@mui/icons-material/Refresh";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import EditNoteOutlinedIcon from "@mui/icons-material/EditNoteOutlined";

import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteDocumentByTitle,
  fetchChunksByDocId,
  fetchDocuments,
  fetchPreviewUrlByDocId,
  updateChunkText,
} from "../api/documents";

import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";

type DocRow = {
  sha256: string;
  title: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
  updated_at: string;
};

const DOC_LIMIT = 50;
const CHUNK_LIMIT = 50;

export default function DocumentListPage() {
  const nav = useNavigate();
  const qc = useQueryClient();

  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<DocRow | null>(null);
  const [tab, setTab] = useState<0 | 1>(0);
  const [chunkEditMode, setChunkEditMode] = useState(false);
  const [chunkDrafts, setChunkDrafts] = useState<Record<string, string>>({});

  const [docOffset] = useState(0);
  const [chunkOffset, setChunkOffset] = useState(0);

  // ✅ 삭제 확인용 state는 반드시 컴포넌트 내부에
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingDeleteTitle, setPendingDeleteTitle] = useState<string | null>(null);

  const docsQ = useQuery({
    queryKey: ["documents", DOC_LIMIT, docOffset],
    queryFn: () => fetchDocuments(DOC_LIMIT, docOffset),
  });

  const docs = useMemo(() => {
    const rows = (docsQ.data?.documents ?? []) as DocRow[];
    if (!search.trim()) return rows;
    const q = search.trim().toLowerCase();
    return rows.filter((d) => (d.title || "").toLowerCase().includes(q));
  }, [docsQ.data, search]);

  const delM = useMutation({
    mutationFn: (title: string) => deleteDocumentByTitle(title),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["documents"] });
      setSelected(null);
    },
  });

  const chunksQ = useQuery({
    enabled: !!selected?.sha256 && tab === 1,
    queryKey: ["chunks", selected?.sha256, CHUNK_LIMIT, chunkOffset],
    queryFn: () => fetchChunksByDocId(selected!.sha256, CHUNK_LIMIT, chunkOffset),
  });

  const previewQ = useQuery({
    enabled: !!selected?.sha256 && tab === 0 && selected?.mime_type === "application/pdf",
    queryKey: ["preview-url", selected?.sha256],
    queryFn: () => fetchPreviewUrlByDocId(selected!.sha256, 3600),
  });

  const previewUrl = previewQ.data?.preview_url;

  const chunkUpdateM = useMutation({
    mutationFn: ({ chunkId, text }: { chunkId: string; text: string }) => updateChunkText(chunkId, text),
    onSuccess: async () => {
      await chunksQ.refetch();
    },
  });

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Typography sx={{ fontSize: 22, fontWeight: 800 }}>문서 관리</Typography>

        <Box sx={{ display: "flex", gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<UploadFileIcon />}
            sx={{ borderRadius: 999 }}
            onClick={() => nav("/parse")}
          >
            파일 등록
          </Button>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            sx={{ borderRadius: 2 }}
            onClick={() => docsQ.refetch()}
          >
            새로고침
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: "flex", gap: 1.5, alignItems: "center", mb: 2 }}>
        <TextField
          size="small"
          placeholder="파일 검색"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          sx={{ width: 340 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon sx={{ color: "#9ca3af" }} />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      <Box sx={{ display: "grid", gridTemplateColumns: "420px 1fr", gap: 2 }}>
        {/* Left list */}
        <Box sx={{ border: "1px solid #eef0f3", borderRadius: 3, overflow: "hidden" }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 90px",
              bgcolor: "#dbeafe",
              px: 1.5,
              py: 1,
            }}
          >
            <Typography sx={{ fontWeight: 800, fontSize: 13 }}>파일 이름</Typography>
            <Typography sx={{ fontWeight: 800, fontSize: 13, textAlign: "center" }}>관리</Typography>
          </Box>

          <Box sx={{ maxHeight: 520, overflow: "auto" }}>
            {docs.map((d) => {
              const active = selected?.sha256 === d.sha256;
              return (
                <Box
                  key={d.sha256}
                  onClick={() => {
                    setSelected(d);
                    setTab(0);
                    setChunkOffset(0);
                  }}
                  sx={{
                    display: "grid",
                    gridTemplateColumns: "1fr 90px",
                    alignItems: "center",
                    px: 1.5,
                    py: 1.1,
                    borderBottom: "1px solid #f1f5f9",
                    cursor: "pointer",
                    bgcolor: active ? "#eef2ff" : "#fff",
                    "&:hover": { bgcolor: active ? "#eef2ff" : "#f9fafb" },
                  }}
                >
                  <Box>
                    <Typography sx={{ fontWeight: 700, fontSize: 14, lineHeight: 1.2 }}>
                      {d.title}
                    </Typography>
                    <Typography sx={{ fontSize: 12, color: "#6b7280", mt: 0.2 }}>
                      {d.mime_type} · {(d.size_bytes / 1024).toFixed(1)} KB
                    </Typography>
                  </Box>

                  <Box sx={{ display: "flex", justifyContent: "center" }}>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        // ✅ 바로 삭제하지 말고 confirm 띄우기
                        setPendingDeleteTitle(d.title);
                        setConfirmOpen(true);
                      }}
                      sx={{ color: "#ef4444" }}
                    >
                      <DeleteOutlineIcon />
                    </IconButton>
                  </Box>
                </Box>
              );
            })}

            {!docs.length && (
              <Typography sx={{ p: 2, color: "#6b7280", textAlign: "center" }}>
                문서가 없습니다. (Parse에서 업로드하세요)
              </Typography>
            )}
          </Box>
        </Box>

        {/* Right panel */}
        <Box sx={{ border: "1px solid #eef0f3", borderRadius: 3, overflow: "hidden", minHeight: 560 }}>
          <Box sx={{ px: 2, py: 1.5 }}>
            <Typography sx={{ fontWeight: 800, fontSize: 16 }}>
              {selected ? selected.title : "문서를 선택하세요"}
            </Typography>
            {selected && (
              <Typography sx={{ color: "#6b7280", fontSize: 12, mt: 0.3 }}>
                doc_id(sha256): {selected.sha256}
              </Typography>
            )}
          </Box>
          <Divider />

          <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ px: 1.5 }}>
            <Tab label="미리보기" />
            <Tab label="Chunks" />
          </Tabs>
          <Divider />

          <Box sx={{ p: 2 }}>
            {!selected && (
              <Typography sx={{ color: "#6b7280", textAlign: "center", mt: 8 }}>
                좌측에서 문서를 선택해주세요.
              </Typography>
            )}

            {selected && tab === 0 && (
              <Box>
                {selected.mime_type === "application/pdf" ? (
                  <Box>
                    <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
                      {previewUrl && (
                        <Button
                          size="small"
                          variant="outlined"
                          startIcon={<OpenInNewIcon />}
                          onClick={() => window.open(previewUrl, "_blank")}
                          sx={{ borderRadius: 2 }}
                        >
                          새 탭에서 열기
                        </Button>
                      )}
                    </Box>

                    {previewUrl ? (
                      <Box
                        sx={{
                          border: "1px solid #eef0f3",
                          borderRadius: 2,
                          overflow: "hidden",
                          height: 460,
                        }}
                      >
                        <iframe
                          title="pdf-preview"
                          src={previewUrl}
                          style={{ width: "100%", height: "100%", border: 0 }}
                        />
                      </Box>
                    ) : (
                      <Typography sx={{ color: "#6b7280" }}>
                        미리보기 URL을 불러오는 중…
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <Typography sx={{ color: "#6b7280" }}>
                    현재 PDF 미리보기에 최적화되어 있습니다. (mime: {selected.mime_type})
                  </Typography>
                )}
              </Box>
            )}

            {selected && tab === 1 && (
              <Box>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                  <Typography sx={{ fontWeight: 800, fontSize: 14 }}>
                    Chunks (총 {chunksQ.data?.metadata?.total ?? chunksQ.data?.metadata?.count ?? 0})
                  </Typography>

                  <Box sx={{ display: "flex", gap: 1 }}>
                    <Button
                      size="small"
                      variant={chunkEditMode ? "contained" : "outlined"}
                      startIcon={<EditNoteOutlinedIcon />}
                      sx={{ borderRadius: 2 }}
                      onClick={() => {
                        setChunkEditMode((v) => !v);
                        setChunkDrafts({});
                      }}
                    >
                      {chunkEditMode ? "편집 종료" : "편집 모드"}
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      sx={{ borderRadius: 2 }}
                      disabled={chunkOffset <= 0}
                      onClick={() => setChunkOffset((v) => Math.max(0, v - CHUNK_LIMIT))}
                    >
                      이전
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      sx={{ borderRadius: 2 }}
                      disabled={(chunksQ.data?.chunks?.length ?? 0) < CHUNK_LIMIT}
                      onClick={() => setChunkOffset((v) => v + CHUNK_LIMIT)}
                    >
                      다음
                    </Button>
                  </Box>
                </Box>

                <Box sx={{ display: "grid", gap: 1 }}>
                  {(chunksQ.data?.chunks ?? []).map((c) => (
                    <Box
                      key={c.id}
                      sx={{
                        border: "1px solid #eef0f3",
                        borderRadius: 2,
                        p: 1.5,
                        "&:hover": { bgcolor: "#f9fafb" },
                      }}
                    >
                      {chunkEditMode ? (
                        <Box sx={{ display: "grid", gap: 1 }}>
                          <TextField
                            multiline
                            minRows={3}
                            maxRows={12}
                            value={chunkDrafts[c.chunk_id] ?? c.text}
                            onChange={(e) =>
                              setChunkDrafts((prev) => ({
                                ...prev,
                                [c.chunk_id]: e.target.value,
                              }))
                            }
                            sx={{ "& .MuiInputBase-root": { fontSize: 14 } }}
                          />
                          <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                            <Button
                              size="small"
                              variant="contained"
                              disabled={chunkUpdateM.isPending}
                              onClick={() => {
                                const nextText = (chunkDrafts[c.chunk_id] ?? c.text).trim();
                                if (!nextText || nextText === c.text) return;
                                chunkUpdateM.mutate({ chunkId: c.chunk_id, text: nextText });
                              }}
                            >
                              저장
                            </Button>
                          </Box>
                        </Box>
                      ) : (
                        <Typography sx={{ fontSize: 14, mt: 0.2, whiteSpace: "pre-wrap" }}>
                          {c.text}
                        </Typography>
                      )}
                    </Box>
                  ))}

                  {chunksQ.isLoading && <Typography sx={{ color: "#6b7280" }}>Chunks 로딩 중…</Typography>}
                  {chunksQ.isError && <Typography sx={{ color: "#ef4444" }}>Chunks 로드 실패</Typography>}
                  {chunkUpdateM.isError && <Typography sx={{ color: "#ef4444" }}>Chunk 저장 실패</Typography>}
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      {/* ✅ Delete confirm dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle sx={{ fontWeight: 800 }}>정말 삭제하시겠습니까?</DialogTitle>
        <DialogContent>
          <Typography sx={{ color: "#6b7280", mt: 0.5 }}>
            {pendingDeleteTitle ?? "-"} 문서를 삭제합니다. (PostgreSQL/OpenSearch/MinIO에서 함께 제거될 수 있습니다)
          </Typography>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button
            variant="outlined"
            sx={{ borderRadius: 2 }}
            onClick={() => {
              setConfirmOpen(false);
              setPendingDeleteTitle(null);
            }}
          >
            취소
          </Button>
          <Button
            variant="contained"
            color="error"
            sx={{ borderRadius: 2 }}
            onClick={() => {
              if (pendingDeleteTitle) delM.mutate(pendingDeleteTitle);
              setConfirmOpen(false);
              setPendingDeleteTitle(null);
            }}
            disabled={delM.isPending}
          >
            삭제
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
