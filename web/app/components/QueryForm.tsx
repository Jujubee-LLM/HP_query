"use client";

import { useMemo, useRef, useState } from "react";
import EvidencePanel from "./EvidencePanel";
import type { Mode, RetrievalMeta, UsedChunk } from "./types";

type ChatRole = "user" | "assistant";
type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  createdAt: number;
  usedChunks?: UsedChunk[];
  retrieval?: RetrievalMeta | null;
  error?: string | null;
};

export default function QueryForm() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("rag");
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);

  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000", []);
  const listRef = useRef<HTMLDivElement | null>(null);

  function sanitizeForLiveDisplay(text: string): string {
    return String(text || "");
  }

  function scrollToBottom() {
    const el = listRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }

  function newId(prefix: string) {
    return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  function resetChat() {
    setMessages([]);
    setActiveMessageId(null);
    setActiveChunkId(null);
    setError(null);
    setIsStreaming(false);
    setLoading(false);
  }

  async function copyActiveAnswer() {
    const text = (activeAssistant?.text || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore
    }
  }

  async function onAsk(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;

    setError(null);
    setLoading(true);
    setIsStreaming(true);

    const userMsg: ChatMessage = { id: newId("u"), role: "user", text: q, createdAt: Date.now() };
    const assistantId = newId("a");
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      text: "",
      createdAt: Date.now(),
      usedChunks: [],
      retrieval: null,
      error: null
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setActiveMessageId(assistantId);
    setActiveChunkId(null);
    setQuestion("");
    setTimeout(scrollToBottom, 0);

    try {
      if (mode === "retrieve_only") {
        const res = await fetch(`${apiBase}/query`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, mode: "retrieve_only" })
        });

        if (res.status === 401) {
          window.location.href = "/login?next=/";
          return;
        }
        if (!res.ok) throw new Error(await res.text());

        const data = (await res.json()) as any;
        const finalAnswer = String(data?.answer || "");
        const used = (data?.used_chunks || []) as UsedChunk[];
        const retrieval = (data?.retrieval || null) as RetrievalMeta | null;

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, text: finalAnswer, usedChunks: used, retrieval } : m))
        );
        if (used[0]) setActiveChunkId(used[0].chunk_id);
      } else {
        const res = await fetch(`${apiBase}/query/stream`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, mode: "rag" })
        });

        if (res.status === 401) {
          window.location.href = "/login?next=/";
          return;
        }
        if (!res.ok) throw new Error(await res.text());

        const reader = res.body?.getReader();
        if (!reader) throw new Error("stream not supported");

        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let fullAnswer = "";

        const handleEvent = (eventName: string, dataStr: string) => {
          const data = dataStr ? JSON.parse(dataStr) : null;
          if (eventName === "error") {
            const msg = String(data?.error || "stream_failed");
            setError(msg);
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, error: msg } : m)));
            throw new Error(msg);
          }
          if (eventName === "delta") {
            const piece = String(data?.text || "");
            fullAnswer += piece;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: sanitizeForLiveDisplay(fullAnswer) } : m))
            );
            setTimeout(scrollToBottom, 0);
            return;
          }
          if (eventName === "meta") {
            if (data?.status === "retrieved") {
              const used = (data?.used_chunks || []) as UsedChunk[];
              const retrieval = (data?.retrieval || null) as RetrievalMeta | null;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, usedChunks: used, retrieval } : m))
              );
              if (used[0]) setActiveChunkId(used[0].chunk_id);
            }
            return;
          }
          if (eventName === "done") {
            const finalAnswer = String(data?.answer || fullAnswer);
            const used = (data?.used_chunks || []) as UsedChunk[];
            const retrieval = (data?.retrieval || null) as RetrievalMeta | null;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: finalAnswer, usedChunks: used, retrieval } : m))
            );
            if (used[0]) setActiveChunkId(used[0].chunk_id);
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          while (true) {
            const sep = buffer.indexOf("\n\n");
            if (sep === -1) break;
            const raw = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);

            let eventName = "message";
            let dataStr = "";
            for (const line of raw.split("\n")) {
              if (line.startsWith("event:")) eventName = line.slice(6).trim();
              else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
            }
            if (eventName) handleEvent(eventName, dataStr);
          }
        }
      }
    } catch (e: any) {
      const msg = e?.message || "request failed";
      setError(msg);
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, error: msg } : m)));
    } finally {
      setLoading(false);
      setIsStreaming(false);
    }
  }

  const activeAssistant = useMemo(() => {
    const id = activeMessageId;
    if (!id) return null;
    const m = messages.find((x) => x.id === id);
    if (!m || m.role !== "assistant") return null;
    return m;
  }, [messages, activeMessageId]);

  const activeChunks = activeAssistant?.usedChunks || [];

  const placeholderExamples = useMemo(() => {
    const examples = [
      "斯内普为什么对哈利态度复杂？",
      "哈利为什么必须住在姨妈家？",
      "魂器为什么这么难毁掉？",
      "分院帽更看重的是天赋还是选择？",
      "伏地魔为什么害怕邓布利多？",
      "霍格沃茨是什么地方？",
      "凤凰社成立的背景是什么？"
    ];
    const pick = () => examples[Math.floor(Math.random() * examples.length)];
    const uniq = new Set<string>();
    const target = Math.floor(Math.random() * 4) + 3; // 3..6
    while (uniq.size < target) uniq.add(pick());
    return `例如：${Array.from(uniq).join(" / ")}`;
  }, []);

  return (
    <div className="grid">
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
          <div>
            <div className="label">提问工作区</div>
            <div className="small">把问题交给羽毛笔。</div>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button className="btn inline" type="button" onClick={copyActiveAnswer} disabled={!activeAssistant?.text?.trim()}>
              复制回答
            </button>
            <button className="btn inline" type="button" onClick={resetChat} disabled={loading || messages.length === 0}>
              新对话
            </button>
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <form onSubmit={onAsk}>
            <div style={{ display: "grid", gap: 10 }}>
              <div className="row">
                <div>
                  <div className="label">模式</div>
                  <select
                    className="input"
                    value={mode}
                    onChange={(e) => setMode(e.target.value as Mode)}
                    disabled={loading}
                  >
                    <option value="rag">问答</option>
                    <option value="retrieve_only">仅检索</option>
                  </select>
                </div>
                <div>
                  <div className="label">发送</div>
                  <button className="btn primary" disabled={loading || !question.trim()}>
                    {loading ? "查阅中…" : "发送"}
                  </button>
                </div>
              </div>

              <textarea
                className="textarea"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={placeholderExamples}
              />

              {error ? <div className="small" style={{ color: "var(--danger)" }}>{error}</div> : null}
            </div>
          </form>
        </div>

        <div style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
            <div className="label">对话</div>
            <div className="small">{isStreaming ? "书写中…" : "点击任意回答可切换右侧对照段落"}</div>
          </div>

          <div
            ref={listRef}
            className="card"
            style={{ marginTop: 8, padding: 12, maxHeight: "56vh", overflow: "auto" }}
          >
            {messages.length === 0 ? (
              <div className="small">（先写下一个问题吧）</div>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {messages.map((m) => {
                  const isAssistant = m.role === "assistant";
                  const isActive = isAssistant && m.id === activeMessageId;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => (isAssistant ? setActiveMessageId(m.id) : null)}
                      className={`chunk ${isAssistant ? "chatAssistant" : ""} ${isActive ? "active" : ""}`}
                      style={{
                        textAlign: "left",
                        cursor: isAssistant ? "pointer" : "default",
                        borderColor: isActive ? "color-mix(in srgb, var(--accent) 70%, var(--border))" : undefined
                      }}
                    >
                      <div
                        className="small"
                        style={{
                          marginBottom: 6,
                          color: isAssistant ? "var(--muted)" : undefined,
                        }}
                      >
                        {isAssistant ? "羽毛笔" : "你"}
                      </div>
                      <div
                        className={isAssistant ? "assistantBody" : undefined}
                        style={{
                          whiteSpace: "pre-wrap",
                          lineHeight: 1.65,
                        }}
                      >
                        {m.text || (isAssistant ? "…" : "")}
                      </div>
                      {m.error ? <div className="small" style={{ color: "var(--danger)", marginTop: 8 }}>{m.error}</div> : null}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {activeAssistant?.retrieval ? (
            <div className="small" style={{ marginTop: 10 }}>
              命中段落：{activeAssistant.retrieval.kept_k}/{activeAssistant.retrieval.retrieved_k}
              {activeAssistant.retrieval.method ? ` · ${activeAssistant.retrieval.method}` : ""}
            </div>
          ) : null}
        </div>
      </div>

      <EvidencePanel chunks={activeChunks} activeChunkId={activeChunkId} onSelect={(id) => setActiveChunkId(id)} />
    </div>
  );
}
