"use client";

import type { UsedChunk } from "./types";

type Token =
  | { kind: "text"; value: string }
  | { kind: "cite"; chunkId: string };

function tokenize(answer: string): Token[] {
  const re = /\[([A-Za-z0-9_:-]+)\]/g;
  const out: Token[] = [];
  let last = 0;
  let m: RegExpExecArray | null = null;
  while ((m = re.exec(answer)) !== null) {
    const start = m.index;
    if (start > last) out.push({ kind: "text", value: answer.slice(last, start) });
    out.push({ kind: "cite", chunkId: m[1] });
    last = start + m[0].length;
  }
  if (last < answer.length) out.push({ kind: "text", value: answer.slice(last) });
  return out;
}

export default function CitedAnswer(props: {
  answer: string;
  usedChunks: UsedChunk[];
  onSelectChunk: (id: string) => void;
}) {
  const idToChunk = new Map(props.usedChunks.map((c) => [c.chunk_id, c] as const));
  const tokens = tokenize(props.answer || "");

  const order: string[] = [];
  const numById = new Map<string, number>();
  for (const t of tokens) {
    if (t.kind !== "cite") continue;
    if (!numById.has(t.chunkId)) {
      order.push(t.chunkId);
      numById.set(t.chunkId, order.length);
    }
  }

  return (
    <div>
      <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.55 }}>
        {tokens.map((t, i) => {
          if (t.kind === "text") return <span key={i}>{t.value}</span>;
          const n = numById.get(t.chunkId) || 0;
          const known = idToChunk.has(t.chunkId);
          const title = known
            ? `引用 ${n} · p${idToChunk.get(t.chunkId)!.page_start} · ${idToChunk.get(t.chunkId)!.chapter || "无章节名"} · ${idToChunk
                .get(t.chunkId)!
                .score.toFixed(3)}`
            : `未知引用`;
          return (
            <button
              key={i}
              className="footnote"
              style={{ marginLeft: 4 }}
              title={title}
              onClick={() => props.onSelectChunk(t.chunkId)}
            >
              {n || "?"}
            </button>
          );
        })}
      </div>
    </div>
  );
}
