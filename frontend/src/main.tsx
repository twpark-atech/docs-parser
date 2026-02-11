import React from "react";
import ReactDOM from "react-dom/client";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./app/App";
import { GlobalStatusProvider } from "./app/layout/GlobalStatusProvider";

const theme = createTheme({ /* 생략 */ });

const qc = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={qc}>
        <GlobalStatusProvider>
          <App />
        </GlobalStatusProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>
);
