import json
import os
from datetime import datetime, timezone


def append_audit_log(
    audit_log_path: str,
    username: str,
    query: str,
    mode: str,
    top_k: int,
    chunk_ids: list[str],
    retrieval: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "chunk_ids": chunk_ids,
        "retrieval": retrieval or {},
    }
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
