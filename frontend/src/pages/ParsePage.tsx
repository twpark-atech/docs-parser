import { Box, Button, LinearProgress, Typography } from "@mui/material";
import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { parseDocument } from "../api/documents";
import { useNavigate } from "react-router-dom";

type UploadItem = {
  id: string;
  file: File;
  progress: number;
  status: "READY" | "UPLOADING" | "DONE" | "ERROR";
  error?: string;
};

export default function ParsePage() {
  const nav = useNavigate();
  const [items, setItems] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState(false);

  const onDrop = useCallback((accepted: File[]) => {
    const next: UploadItem[] = accepted.map((f) => ({
      id: `${f.name}-${f.size}-${f.lastModified}`,
      file: f,
      progress: 0,
      status: "READY",
    }));

    setItems((prev) => {
      const map = new Map(prev.map((p) => [p.id, p]));
      for (const n of next) map.set(n.id, n);
      return Array.from(map.values());
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
  });

  const startUpload = useCallback(async () => {
    if (!items.length) return;
    setBusy(true);

    try {
      for (const item of items) {
        if (item.status === "DONE") continue;

        setItems((prev) =>
          prev.map((p) => (p.id === item.id ? { ...p, status: "UPLOADING", progress: 0, error: undefined } : p))
        );

        try {
          await parseDocument(item.file, (pct) => {
            setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, progress: pct } : p)));
          });

          setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: "DONE", progress: 100 } : p)));
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: "ERROR", error: msg } : p)));
        }
      }
    } finally {
      setBusy(false);
    }
  }, [items]);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const doneCount = useMemo(() => items.filter((i) => i.status === "DONE").length, [items]);

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Typography sx={{ fontSize: 22, fontWeight: 800 }}>문서 파싱 업로드</Typography>
        <Button variant="outlined" sx={{ borderRadius: 2 }} onClick={() => nav("/documentList")}>
          문서관리로 이동
        </Button>
      </Box>

      <Box
        {...getRootProps()}
        sx={{
          border: "2px dashed #cbd5e1",
          borderRadius: 3,
          p: 4,
          textAlign: "center",
          bgcolor: isDragActive ? "#eef2ff" : "#fff",
          cursor: "pointer",
          mb: 2,
        }}
      >
        <input {...getInputProps()} />
        <UploadFileIcon sx={{ fontSize: 44, color: "#2563eb" }} />
        <Typography sx={{ fontWeight: 800, mt: 1 }}>파일을 드래그&드롭 하거나 클릭해서 선택</Typography>
        <Typography sx={{ color: "#6b7280", mt: 0.5, fontSize: 12 }}>
          지원: PDF, DOCX · 다중 업로드 가능
        </Typography>
      </Box>

      <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 2 }}>
        <Button variant="contained" sx={{ borderRadius: 2 }} disabled={!items.length || busy} onClick={startUpload}>
          파싱 시작
        </Button>
        <Typography sx={{ color: "#6b7280", fontSize: 13 }}>
          {doneCount}/{items.length} 완료
        </Typography>
      </Box>

      <Box sx={{ display: "grid", gap: 1 }}>
        {items.map((it) => (
          <Box
            key={it.id}
            sx={{
              border: "1px solid #eef0f3",
              borderRadius: 2,
              p: 1.5,
              display: "grid",
              gridTemplateColumns: "1fr 160px 40px",
              alignItems: "center",
              gap: 1.5,
            }}
          >
            <Box>
              <Typography sx={{ fontWeight: 800, fontSize: 14 }}>{it.file.name}</Typography>
              <Typography sx={{ color: "#6b7280", fontSize: 12 }}>
                {(it.file.size / 1024).toFixed(1)} KB · {it.status}
              </Typography>
              {it.error && <Typography sx={{ color: "#ef4444", fontSize: 12, mt: 0.5 }}>{it.error}</Typography>}
            </Box>

            <Box>
              <LinearProgress variant="determinate" value={it.progress} />
              <Typography sx={{ fontSize: 12, color: "#6b7280", mt: 0.5 }}>{it.progress}%</Typography>
            </Box>

            <Button
              variant="text"
              onClick={() => removeItem(it.id)}
              disabled={busy && it.status === "UPLOADING"}
              sx={{ minWidth: 0 }}
            >
              <DeleteOutlineIcon sx={{ color: "#ef4444" }} />
            </Button>
          </Box>
        ))}

        {!items.length && (
          <Typography sx={{ color: "#6b7280", textAlign: "center", mt: 3 }}>업로드할 파일을 추가해주세요.</Typography>
        )}
      </Box>
    </Box>
  );
}
