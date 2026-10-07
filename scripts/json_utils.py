"""Reusable JSON-safe conversion and atomic JSON writes."""
import json
import os
from pathlib import Path


def to_json_safe(value):
    """Recursively convert numpy scalars/arrays to native JSON types.

    Unsupported types raise TypeError so real bugs are not hidden.
    """
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return [to_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_json_safe(data), ensure_ascii=False, indent=2)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
