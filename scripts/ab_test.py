import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _local_paths_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    storage = repo_root / "storage"
    return {
        "CHUNKS_PATH": str(storage / "chunks.jsonl"),
        "FAISS_DIR": str(storage / "faiss"),
        "ENTITY_INDEX_PATH": str(storage / "entity_index.json"),
        "ALIASES_PATH": str(storage / "aliases.json"),
        "AUDIT_LOG_PATH": str(storage / "logs.jsonl"),
        "DATA_PDF_PATH": str(repo_root / "data" / "book.pdf"),
    }


def run_eval(questions: str, report_dir: str, env_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(_local_paths_env())
    env.update(env_overrides)
    os.makedirs(report_dir, exist_ok=True)
    t0 = time.time()
    subprocess.check_call(
        [sys.executable, "-B", os.path.join("scripts", "eval.py"), "--questions", questions, "--report_dir", report_dir],
        env=env,
    )
    dt = time.time() - t0
    report_path = os.path.join(report_dir, "report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["elapsed_sec"] = round(dt, 2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="scripts/questions.jsonl")
    ap.add_argument("--report_root", default="storage/reports/ab_test")
    args = ap.parse_args()

    configs = [
        (
            "A_no_expand_no_rerank",
            {
                "RETRIEVAL_EXPAND_ENABLED": "false",
                "RETRIEVAL_RERANK_ENABLED": "false",
            },
        ),
        (
            "B_expand_only",
            {
                "RETRIEVAL_EXPAND_ENABLED": "true",
                "RETRIEVAL_EXPAND_N": os.environ.get("RETRIEVAL_EXPAND_N", "2"),
                "RETRIEVAL_RERANK_ENABLED": "false",
            },
        ),
        (
            "C_expand_rerank",
            {
                "RETRIEVAL_EXPAND_ENABLED": "true",
                "RETRIEVAL_EXPAND_N": os.environ.get("RETRIEVAL_EXPAND_N", "2"),
                "RETRIEVAL_RERANK_ENABLED": "true",
                "RETRIEVAL_RERANK_CANDIDATES": os.environ.get("RETRIEVAL_RERANK_CANDIDATES", "24"),
                "RETRIEVAL_RERANK_MAX_CHARS": os.environ.get("RETRIEVAL_RERANK_MAX_CHARS", "800"),
            },
        ),
    ]

    os.makedirs(args.report_root, exist_ok=True)
    summary = []
    for name, env_overrides in configs:
        report_dir = os.path.join(args.report_root, name)
        report = run_eval(args.questions, report_dir, env_overrides)
        summary.append({"name": name, **report})

    summary_path = os.path.join(args.report_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("A/B summary:")
    for row in summary:
        print(
            f"- {row['name']}: hit_rate@k={row['hit_rate@k']:.3f}, "
            f"MRR={row['MRR']:.3f}, citation_coverage={row['citation_coverage']:.3f}, "
            f"unsupported_claim_rate={row['unsupported_claim_rate']:.3f}, "
            f"elapsed_sec={row['elapsed_sec']}"
        )
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
