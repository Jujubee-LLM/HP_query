"use client";

import { useMemo, useState } from "react";
import type { UsedChunk } from "./types";

export default function EvidencePanel(props: {
  chunks: UsedChunk[];
  activeChunkId: string | null;
  onSelect: (id: string) => void;
}) {
  const [showDetails, setShowDetails] = useState(false);

  const rows = useMemo(() => {
    // Keep backend order stable so it matches "来源#"/"[来源#]" in the answer prompt.
    return [...props.chunks];
  }, [props.chunks]);

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
        <div>
          <div className="label">对照段落</div>
          <div className="small">可点击条目查看摘要，必要时展开细节。</div>
        </div>
        <span className="badge">条目: {rows.length}</span>
      </div>

      <div style={{ marginTop: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <label className="small" style={{ display: "flex", alignItems: "center", gap: 8, userSelect: "none" }}>
          <input type="checkbox" checked={showDetails} onChange={(e) => setShowDetails(e.target.checked)} />
          显示细节
        </label>
      </div>

      <div style={{ display: "grid", gap: 8, marginTop: 12 }}>
        {rows.length === 0 ? (
          <div className="small">（回答后这里会出现相关段落）</div>
        ) : null}
        {rows.map((c, idx) => {
          const active = props.activeChunkId === c.chunk_id;
          return (
            <button
              key={c.chunk_id}
              className={`chunk ${active ? "active" : ""}`}
              style={{ textAlign: "left", cursor: "pointer" }}
              onClick={() => props.onSelect(c.chunk_id)}
            >
              <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
                <span className="badge">来源{idx + 1}</span>
                <div style={{ display: "grid", gap: 6, width: "100%" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
                    <div className="small" style={{ color: "var(--text)" }}>
                      p{c.page_start}
                      {c.page_end !== c.page_start ? `–${c.page_end}` : ""}
                      {c.chapter ? ` · ${c.chapter}` : ""}
                    </div>
                    <div className="small">
                      {c.score.toFixed(3)}
                      {showDetails && c.rerank_score != null ? ` · rerank ${c.rerank_score.toFixed(3)}` : ""}
                      {showDetails && c.vector_score != null ? ` · vec ${c.vector_score.toFixed(3)}` : ""}
                      {showDetails && c.bm25_score != null ? ` · bm25 ${c.bm25_score.toFixed(3)}` : ""}
                    </div>
                  </div>
                  <div style={{ color: "var(--text)", lineHeight: 1.55 }}>{c.text_snippet}</div>
                  {showDetails ? (
                    <div className="mono small" style={{ wordBreak: "break-all" }}>
                      {c.chunk_id}
                    </div>
                  ) : null}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
