"use client";

import { useMemo, useState } from "react";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const nextPath = useMemo(() => {
    if (typeof window === "undefined") return "/";
    const u = new URL(window.location.href);
    return u.searchParams.get("next") || "/";
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
      const res = await fetch(`${base}/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "login failed");
      }
      window.location.href = nextPath;
    } catch (e: any) {
      setErr(e?.message || "login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card" style={{ maxWidth: 520, margin: "40px auto" }}>
      <h1 style={{ marginTop: 0, marginBottom: 6 }}>进入自习室</h1>
      <p className="small">请输入账号密码。</p>
      <form onSubmit={onSubmit}>
        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <div className="label">用户名</div>
            <input className="input" value={username} onChange={(e) => setUsername(e.target.value)} />
          </div>
          <div>
            <div className="label">密码</div>
            <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          {err ? <div className="small" style={{ color: "#fca5a5" }}>{err}</div> : null}
          <button className="btn primary" disabled={loading}>
            {loading ? "校验中…" : "进入"}
          </button>
        </div>
      </form>
    </div>
  );
}
