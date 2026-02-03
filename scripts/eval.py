import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable


def _ensure_import_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(repo_root, "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    # Ensure local paths when running outside Docker
    os.environ.setdefault("CHUNKS_PATH", os.path.join(repo_root, "storage", "chunks.jsonl"))
    os.environ.setdefault("FAISS_DIR", os.path.join(repo_root, "storage", "faiss"))
    os.environ.setdefault("ENTITY_INDEX_PATH", os.path.join(repo_root, "storage", "entity_index.json"))
    os.environ.setdefault("ALIASES_PATH", os.path.join(repo_root, "storage", "aliases.json"))
    os.environ.setdefault("AUDIT_LOG_PATH", os.path.join(repo_root, "storage", "logs.jsonl"))
    os.environ.setdefault("DATA_PDF_PATH", os.path.join(repo_root, "data", "book.pdf"))


_ensure_import_path()

from app.retrieval import retrieve  # noqa: E402
from app.rag import rag_answer, retrieve_only_answer  # noqa: E402


_CIT_RE = re.compile(r"\[([A-Za-z0-9_:-]+)\]")


def extract_citations(text: str) -> set[str]:
    return set(m.group(1) for m in _CIT_RE.finditer(text or ""))


def sentence_split(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？\\n])", text)
    return [p.strip() for p in parts if p.strip()]


def citation_coverage(answer: str) -> float:
    sents = sentence_split(answer)
    if not sents:
        return 0.0
    with_cit = sum(1 for s in sents if extract_citations(s))
    return with_cit / len(sents)


@dataclass
class Q:
    qid: str
    question: str
    gold_chunk_ids: list[str]
    k: int
    mode: str


def iter_questions(path: str) -> Iterable[Q]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Q(
                qid=obj.get("id", ""),
                question=obj["question"],
                gold_chunk_ids=obj.get("gold_chunk_ids", []),
                k=int(obj.get("k", 8)),
                mode=obj.get("mode", "retrieve_only"),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--report_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    total = 0
    hit = 0
    mrr_sum = 0.0
    cov_sum = 0.0
    unsupported_sum = 0.0

    preds_path = os.path.join(args.report_dir, "predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as pf:
        for q in iter_questions(args.questions):
            total += 1
            used, retrieved_ids, meta = retrieve(q.question, q.k)

            rank = None
            gold = set(q.gold_chunk_ids)
            if gold:
                for i, cid in enumerate(retrieved_ids, start=1):
                    if cid in gold:
                        rank = i
                        break

            if rank is not None:
                hit += 1
                mrr_sum += 1.0 / rank

            if q.mode == "rag":
                answer = rag_answer(q.question, used)
            else:
                answer = retrieve_only_answer(q.question, used)

            cov = citation_coverage(answer)
            cov_sum += cov
            unsupported_sum += (1.0 - cov)

            pf.write(
                json.dumps(
                    {
                        "id": q.qid,
                        "question": q.question,
                        "mode": q.mode,
                        "k": q.k,
                        "retrieved_chunk_ids": retrieved_ids,
                        "gold_chunk_ids": q.gold_chunk_ids,
                        "first_gold_rank": rank,
                        "answer": answer,
                        "citations_in_answer": sorted(list(extract_citations(answer))),
                        "citation_coverage": cov,
                        "unsupported_claim_rate": 1.0 - cov,
                        "retrieval_method": (meta or {}).get("method"),
                        "expand_enabled": (meta or {}).get("expand_enabled"),
                        "rerank_enabled": (meta or {}).get("rerank_enabled"),
                        "expand_queries": (meta or {}).get("expand_queries"),
                    },
                    ensure_ascii=False,
                )
                + "\\n"
            )

    report = {
        "n": total,
        "hit_rate@k": (hit / total) if total else 0.0,
        "MRR": (mrr_sum / total) if total else 0.0,
        "citation_coverage": (cov_sum / total) if total else 0.0,
        "unsupported_claim_rate": (unsupported_sum / total) if total else 0.0,
        "notes": "unsupported_claim_rate 以“句子未包含任何 [chunk_id] 引用”为启发式估计。",
    }

    report_path = os.path.join(args.report_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {report_path}")
    print(f"Wrote: {preds_path}")


if __name__ == "__main__":
    main()
