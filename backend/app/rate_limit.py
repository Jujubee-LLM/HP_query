import json
import os
import threading
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from .settings import settings

_LOCK = threading.Lock()


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _load_state(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def _normalize_state(state: dict) -> dict:
    """
    Keep compatibility with the previous shape where the top-level was `{month: {uid: used}}`.
    New shape:
    {
      "usage": {month: {uid: used_count}},
      "bonus": {month: {uid: extra_allowance}},
      "codes": {...}
    }
    """
    state = state if isinstance(state, dict) else {}
    looks_like_month = lambda k: isinstance(k, str) and len(k) == 7 and k[4] == "-" and k[:4].isdigit() and k[5:7].isdigit()
    preserved_codes = state.get("codes") if isinstance(state.get("codes"), dict) else {}
    if "usage" not in state:
        usage = {k: v for k, v in state.items() if looks_like_month(k) and isinstance(v, dict)}
        state = {"usage": usage, "codes": preserved_codes}
    state.setdefault("bonus", {})
    state.setdefault("codes", {})
    # Coerce core maps to dict to avoid type errors if the file was hand-edited.
    state["usage"] = state.get("usage") if isinstance(state.get("usage"), dict) else {}
    state["bonus"] = state.get("bonus") if isinstance(state.get("bonus"), dict) else {}
    state["codes"] = state.get("codes") if isinstance(state.get("codes"), dict) else {}
    # Drop any month entries accidentally sitting at top-level once normalized.
    return state


def _get_or_set_anon_id(request: Request) -> str:
    sid = request.session.get("anon_id")
    if sid:
        return sid
    sid = f"anon_{int(datetime.now(timezone.utc).timestamp()*1000)}"
    request.session["anon_id"] = sid
    return sid


def _is_exempt(user: dict | None) -> bool:
    raw = (settings.rate_limit_exempt_users or "").strip()
    if not raw or not user:
        return False
    allowed = {u.strip() for u in raw.split(",") if u.strip()}
    return bool(user.get("username") in allowed)


def check_and_increment(request: Request, user: dict | None) -> None:
    if not bool(getattr(settings, "rate_limit_enabled", True)):
        return
    if _is_exempt(user):
        return

    uid = user.get("username") if user else _get_or_set_anon_id(request)
    now = datetime.now(timezone.utc)
    month = _month_key(now)
    limit = int(getattr(settings, "rate_limit_monthly_limit", 30) or 30)
    path = str(getattr(settings, "rate_limit_path", "") or "")

    with _LOCK:
        state = _normalize_state(_load_state(path))

        usage_map = state["usage"].get(month, {}) if isinstance(state.get("usage"), dict) else {}
        bonus_map = state["bonus"].get(month, {}) if isinstance(state.get("bonus"), dict) else {}

        used = int(usage_map.get(uid, 0) or 0)
        bonus = int(bonus_map.get(uid, 0) or 0)
        effective_limit = limit + bonus

        if used >= effective_limit:
            raise HTTPException(status_code=429, detail="已达本月提问上限，请充值")
        usage_map[uid] = used + 1
        state["usage"][month] = usage_map
        state["bonus"][month] = bonus_map
        _save_state(path, state)
