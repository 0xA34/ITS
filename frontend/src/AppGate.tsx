// src/AppGate.tsx
import { useEffect, useState } from "react";
import NotFound from "./pages/NotFound";

export default function AppGate({ children }: { children: React.ReactNode }) {
  const [ok, setOk] = useState<boolean | null>(null);

  useEffect(() => {
    const run = async () => {
      try {
        const res = await fetch("/api/health");
        setOk(res.ok);
      } catch {
        setOk(false);
      }
    };
    run();
  }, []);

  if (ok === null) return <div>Checking server...</div>;
  if (!ok) return <NotFound />;

  return <>{children}</>;
}
