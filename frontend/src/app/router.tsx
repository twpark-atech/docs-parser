import { createBrowserRouter, Navigate } from "react-router-dom";
import AppShell from "./layout/AppShell";

import ConversationsPage from "../pages/ConversationsPage";
import ChatListPage from "../pages/ChatListPage";
import DocumentListPage from "../pages/DocumentListPage";
import ParsePage from "../pages/ParsePage";

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Navigate to="/conversations" replace /> },
      { path: "/conversations", element: <ConversationsPage /> },
      { path: "/chatList", element: <ChatListPage /> },
      { path: "/documentList", element: <DocumentListPage /> },
      { path: "/parse", element: <ParsePage /> },
    ],
  },
]);
