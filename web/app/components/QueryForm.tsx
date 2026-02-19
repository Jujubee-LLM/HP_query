"use client";

import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import EvidencePanel from "./EvidencePanel";
import type { RetrievalMeta, UsedChunk } from "./types";

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

type BlockedRequest = {
  question: string;
  assistantId: string;
};

export default function QueryForm() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);

  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showPaywall, setShowPaywall] = useState(false);
  const [redeemCode, setRedeemCode] = useState("");
  const [redeemMsg, setRedeemMsg] = useState<string | null>(null);
  const [blockedRequest, setBlockedRequest] = useState<BlockedRequest | null>(null);

  const searchParams = useSearchParams();
  const showDebug = (searchParams?.get("debug") || "") === "1";
  const metaClass = showDebug ? "" : "meta-hidden";

  const apiBase = useMemo(() => process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000", []);
  const listRef = useRef<HTMLDivElement | null>(null);
  const paywallTimerRef = useRef<number | null>(null);

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
    setBlockedRequest(null);
  }

  function openPaywall() {
    if (paywallTimerRef.current) window.clearTimeout(paywallTimerRef.current);
    setRedeemMsg(null);
    setRedeemCode("");
    setShowPaywall(true);
  }

  function closePaywall() {
    if (paywallTimerRef.current) window.clearTimeout(paywallTimerRef.current);
    setShowPaywall(false);
    setRedeemMsg(null);
    setRedeemCode("");
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

    void fetchAndStream({ question: q, assistantId });
  }

  async function fetchAndStream(params: { question: string; assistantId: string }) {
    const { question: q, assistantId } = params;
    setError(null);
    setLoading(true);
    setIsStreaming(true);
    setBlockedRequest(null);

    try {
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
      if (res.status === 429) {
        setBlockedRequest({ question: q, assistantId });
        openPaywall();
        throw new Error("RATE_LIMIT");
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
    } catch (e: any) {
      const msg = e?.message || "request failed";
      if (msg !== "RATE_LIMIT") {
        setError(msg);
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, error: msg } : m)));
      }
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

  const examplePack = useMemo(() => {
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
    const items = Array.from(uniq);
    const block = items.length === 0
      ? ""
      : [`例如：${items[0]}`, ...items.slice(1).map((x) => `/ ${x}`)].join("\n");
    return {
      block,
      placeholder: "写下你的问题…"
    };
  }, []);

  const couplet = {
    upper: "霍格沃茨藏千卷",
    lower: "羽毛笔引万般答"
  };

  return (
    <div className="homeLayout">
      {showPaywall ? (
        <div className="modalOverlay" onClick={closePaywall}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="label" style={{ fontSize: 18 }}>已达本月提问上限，请充值</div>
            <div className="small" style={{ marginTop: 6 }}>
              按次数计费，联系站点管理员获取兑换码。暂不支持退款与取消。
            </div>
            <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
              <div className="small">兑换码</div>
              <input
                className="input"
                value={redeemCode}
                onChange={(e) => setRedeemCode(e.target.value)}
                placeholder="输入兑换码"
              />
              {redeemMsg ? <div className="small" style={{ color: "var(--danger)" }}>{redeemMsg}</div> : null}
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
                <button
                  className="btn inline"
                  type="button"
                  onClick={async () => {
                    setRedeemMsg(null);
                    const code = redeemCode.trim();
                    if (!code) {
                      setRedeemMsg("请输入兑换码");
                      return;
                    }
                    try {
                      const res = await fetch(`${apiBase}/quota/redeem`, {
                        method: "POST",
                        credentials: "include",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ code })
                      });
                      if (!res.ok) {
                        const data = await res.json().catch(() => ({}));
                        setRedeemMsg(String(data?.detail || "兑换失败"));
                        return;
                      }
                      setRedeemMsg("兑换成功，可继续提问");
                      setRedeemCode("");
                      if (paywallTimerRef.current) window.clearTimeout(paywallTimerRef.current);
                      paywallTimerRef.current = window.setTimeout(() => closePaywall(), 3000);
                      if (blockedRequest) {
                        try {
                          await fetchAndStream(blockedRequest);
                          setBlockedRequest(null);
                        } catch {
                          // swallow; fetchAndStream already handles errors/flags
                        }
                      }
                    } catch {
                      setRedeemMsg("兑换失败");
                    }
                  }}
                >
                  兑换
                </button>
                <button className="btn inline" type="button" onClick={closePaywall}>
                  关闭
                </button>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
              <span className="small">如需充值，请联系站点管理员。</span>
            </div>
          </div>
        </div>
      ) : null}
      <aside className="couplet coupletLeft">
        <div className="coupletLine">{couplet.upper}</div>
      </aside>

      <section className="mainColumn">
        <div className="mainRow">
          <div className="mainStack">
            <div className="card">
              <div className="askToolbar">
                <button
                  className="btn askActionBtn"
                  type="button"
                  onClick={copyActiveAnswer}
                  disabled={!activeAssistant?.text?.trim()}
                >
                  复制回答
                </button>
                <button
                  className="btn askActionBtn"
                  type="button"
                  onClick={resetChat}
                  disabled={loading || messages.length === 0}
                >
                  新对话
                </button>
              </div>

              <div className="askPanel">
                <div className="askHeader">
                  <div>
                    <div className="askTitleLine">
                      <span className="askTitle">提问工作区</span>
                      <span className="askSubtitle">把问题交给羽毛笔。</span>
                    </div>
                  </div>
                </div>

                <div className="askExamples">{examplePack.block}</div>

                <form onSubmit={onAsk} className="askForm">
                  <div className="askRow">
                    <textarea
                      className="askInput"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder={examplePack.placeholder}
                      rows={1}
                    />
                    <button className="btn askSend" disabled={loading || !question.trim()}>
                      {loading ? "查阅中…" : "发送"}
                    </button>
                  </div>

                  {error ? <div className="small" style={{ color: "var(--danger)" }}>{error}</div> : null}
                </form>
              </div>

              <div style={{ marginTop: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
                  <div className="label">对话</div>
                  <div className="small">
                    {isStreaming ? "书写中…" : showDebug ? "点击任意回答可切换下方对照段落" : ""}
                  </div>
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
                  <div className={`small ${metaClass}`} style={{ marginTop: 10 }}>
                    命中段落：{activeAssistant.retrieval.kept_k}/{activeAssistant.retrieval.retrieved_k}
                    {activeAssistant.retrieval.method ? ` · ${activeAssistant.retrieval.method}` : ""}
                  </div>
                ) : null}
              </div>
            </div>

            {showDebug ? (
              <EvidencePanel
                chunks={activeChunks}
                activeChunkId={activeChunkId}
                onSelect={(id) => setActiveChunkId(id)}
                showDebug={showDebug}
              />
            ) : null}
          </div>

          <aside className="iconRail" aria-label="学院图标">
            <div className="iconTile">
              <img src="/pictures/1.png" alt="图标一" />
            </div>
            <div className="iconTile">
              <img src="/pictures/2.png" alt="图标二" />
            </div>
            <div className="iconTile">
              <img src="/pictures/3.png" alt="图标三" />
            </div>
            <div className="iconTile">
              <img src="/pictures/4.png" alt="图标四" />
            </div>
          </aside>
        </div>
      </section>

      <aside className="couplet coupletRight">
        <div className="coupletLine">{couplet.lower}</div>
      </aside>
    </div>
  );
}
