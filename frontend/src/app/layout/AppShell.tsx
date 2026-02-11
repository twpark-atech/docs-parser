import {
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
} from "@mui/material";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import LogoutIcon from "@mui/icons-material/Logout";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { navItems } from "./nav";
import { useMemo, useState } from "react";
import { useGlobalStatus } from "./GlobalStatusProvider";
import MenuIcon from "@mui/icons-material/Menu";
import Tooltip from "@mui/material/Tooltip";

const SIDEBAR_W = 260;
const SIDEBAR_W_COLLAPSED = 72;

export default function AppShell() {
  const nav = useNavigate();
  const loc = useLocation();
  const { sidebarError } = useGlobalStatus();

  const [collapsed, setCollapsed] = useState(false);
  const sidebarW = collapsed ? SIDEBAR_W_COLLAPSED : SIDEBAR_W;

  const grouped = useMemo(() => {
    const map = new Map<string, typeof navItems>();
    for (const item of navItems) {
      const key = item.group ?? "root";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(item);
    }
    return map;
  }, []);

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
      <Drawer
        variant="permanent"
        PaperProps={{
          sx: {
            width: sidebarW,
            borderRight: "1px solid #eef0f3",
            bgcolor: "#fff",
            px: collapsed ? 0.5 : 1.5,
            py: 2,
            transition: "width 180ms ease",
            overflowX: "hidden",
          },
        }}
      >
        {/* Logo */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: collapsed ? 0.5 : 1,
            mb: 2,
            justifyContent: collapsed ? "center" : "flex-start",
          }}
        >
          <Box sx={{ width: 34, height: 34, borderRadius: 2, bgcolor: "#eef2ff" }} />
          {!collapsed && <Typography sx={{ fontWeight: 700, color: "#111827" }}>ATECH</Typography>}
        </Box>

        {/* Root nav */}
        <List sx={{ px: 0.5 }}>
          {grouped.get("root")?.map((item) => {
            const active = loc.pathname === item.path;
            const Icon = item.icon;
            const isPrimary = item.primary;

            const button = (
              <ListItemButton
                key={item.path}
                onClick={() => nav(item.path)}
                sx={{
                  mb: 0.5,
                  borderRadius: 2,
                  justifyContent: collapsed ? "center" : "flex-start",
                  bgcolor: active && isPrimary ? "primary.main" : active ? "#eef2ff" : "transparent",
                  color: active && isPrimary ? "#fff" : "#111827",
                  "&:hover": { bgcolor: active && isPrimary ? "primary.dark" : "#f3f4f6" },
                  px: collapsed ? 1 : 2,
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: collapsed ? 0 : 40,
                    justifyContent: "center",
                    color: active && isPrimary ? "#fff" : "#2563eb",
                  }}
                >
                  <Icon />
                </ListItemIcon>

                {!collapsed && (
                  <ListItemText primary={item.label} primaryTypographyProps={{ fontWeight: 600 }} />
                )}
              </ListItemButton>
            );

            return collapsed ? (
              <Tooltip key={item.path} title={item.label} placement="right">
                {button}
              </Tooltip>
            ) : (
              <Box key={item.path}>{button}</Box>
            );
          })}
        </List>

        <Divider sx={{ my: 1.5 }} />

        {/* Group title */}
        {!collapsed && (
          <Typography sx={{ px: 1, mb: 1, fontSize: 12, fontWeight: 700, color: "#6b7280" }}>
            지식 자산 관리
          </Typography>
        )}

        {/* Group nav */}
        <List sx={{ px: 0.5 }}>
          {grouped.get("지식 자산 관리")?.map((item) => {
            const active = loc.pathname === item.path;
            const Icon = item.icon;

            const button = (
              <ListItemButton
                key={item.path}
                onClick={() => nav(item.path)}
                sx={{
                  mb: 0.5,
                  borderRadius: 2,
                  justifyContent: collapsed ? "center" : "flex-start",
                  bgcolor: active ? "#eef2ff" : "transparent",
                  "&:hover": { bgcolor: "#f3f4f6" },
                  px: collapsed ? 1 : 2,
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: collapsed ? 0 : 40,
                    justifyContent: "center",
                    color: "#2563eb",
                  }}
                >
                  <Icon />
                </ListItemIcon>

                {!collapsed && (
                  <ListItemText primary={item.label} primaryTypographyProps={{ fontWeight: 600 }} />
                )}
              </ListItemButton>
            );

            return collapsed ? (
              <Tooltip key={item.path} title={item.label} placement="right">
                {button}
              </Tooltip>
            ) : (
              <Box key={item.path}>{button}</Box>
            );
          })}
        </List>

        {/* Footer */}
        <Box sx={{ mt: "auto", px: collapsed ? 0.5 : 1, pb: 1 }}>
          {!collapsed && sidebarError && (
            <Typography sx={{ color: "#dc2626", fontSize: 12, mb: 1 }}>{sidebarError}</Typography>
          )}

          <Button
            fullWidth
            variant="text"
            startIcon={!collapsed ? <LogoutIcon /> : undefined}
            sx={{
              justifyContent: collapsed ? "center" : "flex-start",
              color: "#111827",
              borderRadius: 2,
              minWidth: 0,
              px: collapsed ? 1 : 2,
            }}
          >
            {collapsed ? <LogoutIcon /> : "로그아웃"}
          </Button>
        </Box>
      </Drawer>

      {/* Main */}
      <Box
      sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          ml: `${sidebarW}px`,
          transition: "margin-left 180ms ease",
          minWidth: 0,
      }}
      >
        <Box
        sx={{
            height: 64,
            display: "flex",
            alignItems: "center",
            px: 2,
            gap: 1,
            bgcolor: "#fff",
            borderBottom: "1px solid #eef0f3",
            position: "sticky",
            top: 0,
            zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
        >
          {/* Sidebar toggle */}
          <Tooltip title={collapsed ? "사이드바 펼치기" : "사이드바 접기"}>
            <IconButton onClick={() => setCollapsed((v) => !v)} sx={{ color: "#2563eb" }}>
              <MenuIcon />
            </IconButton>
          </Tooltip>

          {/* Back */}
          <IconButton onClick={() => nav(-1)} sx={{ color: "#2563eb" }}>
            <ChevronLeftIcon />
          </IconButton>

          <Box sx={{ flex: 1, display: "flex", justifyContent: "center" }}>
            <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
              <Box sx={{ width: 22, height: 22, borderRadius: "50%", bgcolor: "#2563eb" }} />
              <Box sx={{ width: 22, height: 22, borderRadius: "50%", bgcolor: "#ef4444" }} />
              <Box sx={{ width: 22, height: 22, borderRadius: "50%", bgcolor: "#f59e0b" }} />
            </Box>
          </Box>

          <Button variant="contained" startIcon={<AccountCircleIcon />} sx={{ borderRadius: 999 }}>
            사용자
          </Button>
        </Box>

        <Box sx={{ p: 2.5 }}>
          <Box
            sx={{
              bgcolor: "#fff",
              borderRadius: 4,
              boxShadow: "0 1px 2px rgba(0,0,0,0.06)",
              minHeight: "calc(100vh - 64px - 40px)",
              p: 3,
            }}
          >
            <Outlet />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
