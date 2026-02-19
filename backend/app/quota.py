import json
import os
import threading
from datetime import datetime, timezone

from fastapi import HTTPException

from .settings import settings
from .rate_limit import _normalize_state

_LOCK = threading.Lock()


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _load(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save(path: str, state: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def redeem_code(code: str, *, user_id: str, anon_id: str | None) -> int:
    code = (code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="兑换码无效")
    path = str(getattr(settings, "rate_limit_path", "") or "")
    if not path:
        raise HTTPException(status_code=500, detail="配置错误")

    with _LOCK:
        state = _normalize_state(_load(path))
        codes = state.get("codes", {}) or {}
        amount = codes.get(code)
        if amount is None:
            raise HTTPException(status_code=400, detail="兑换码无效或已使用")
        try:
            amount = int(amount)
        except Exception:
            raise HTTPException(status_code=400, detail="兑换码无效")
        # consume code
        codes.pop(code, None)
        state["codes"] = codes

        target = user_id or anon_id
        if not target:
            raise HTTPException(status_code=400, detail="无法识别用户")

        month = _month_key(datetime.now(timezone.utc))

        bonus_map_all = state.get("bonus", {}) or {}
        bonus_map = bonus_map_all.get(month, {}) if isinstance(bonus_map_all, dict) else {}
        bonus_map[target] = int(bonus_map.get(target, 0) or 0) + int(amount)

        state["codes"] = codes
        state["bonus"] = bonus_map_all
        state["bonus"][month] = bonus_map

        _save(path, state)
        base_limit = int(getattr(settings, "rate_limit_monthly_limit", 30) or 30)
        return base_limit + int(bonus_map[target])
