import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

type GlobalStatus = {
  sidebarError?: string;
  setSidebarError: (msg?: string) => void;
};

const Ctx = createContext<GlobalStatus | null>(null);

export function GlobalStatusProvider({ children }: { children: React.ReactNode }) {
  const [sidebarError, setSidebarErrorState] = useState<string | undefined>(
    "Failed to load tables. Please try again."
  );

  const setSidebarError = useCallback((msg?: string) => {
    setSidebarErrorState(msg);
  }, []);

  const value = useMemo(() => ({ sidebarError, setSidebarError }), [sidebarError, setSidebarError]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useGlobalStatus() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useGlobalStatus must be used within GlobalStatusProvider");
  return v;
}
