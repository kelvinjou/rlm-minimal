from typing import Any

from realtime import Optional


DEEPSEEK_PRICING_PER_1M = {
    "deepseek-chat": {
        "input_cache_hit": 0.028,
        "input_cache_miss": 0.28,
        "output": 0.42,
    },
    "deepseek-reasoner": {
        "input_cache_hit": 0.028,
        "input_cache_miss": 0.28,
        "output": 0.42,
    },
}


def _to_plain(value: Any):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain(value.dict())
    if hasattr(value, "__dict__"):
        return {key: _to_plain(item) for key, item in vars(value).items()}
    return value


def _to_dict(value: Any) -> dict:
    plain = _to_plain(value)
    if isinstance(plain, dict):
        return plain
    return {}


def _get_nested(data: dict, *keys: str, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
