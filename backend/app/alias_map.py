import json
import os
import threading


_LOCK = threading.Lock()
_CACHE: tuple[str, float, dict[str, list[str]]] | None = None


def load_alias_map(path: str) -> dict[str, list[str]]:
    """
    Load a simple alias map: phrase -> list of canonical entities.
    """
    p = (path or "").strip()
    if not p:
        return {}
    if not os.path.exists(p):
        return {}
    mtime = os.path.getmtime(p)
    global _CACHE
    if _CACHE is not None and _CACHE[0] == p and _CACHE[1] == mtime:
        return _CACHE[2]

    with _LOCK:
        if _CACHE is not None and _CACHE[0] == p and _CACHE[1] == mtime:
            return _CACHE[2]
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            obj = None

        out: dict[str, list[str]] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                if isinstance(v, list):
                    vals = [str(x).strip() for x in v if str(x).strip()]
                    if vals:
                        out[k.strip()] = vals
        _CACHE = (p, mtime, out)
        return out
