"""Utility helpers shared across the project."""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Serialize *data* to *path* with 2-space indentation."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def save_csv_from_list(rows: list, path: Path) -> None:
    """Write a list of dicts as a CSV file."""
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def sanitize_name(name: str) -> str:
    """Replace spaces and non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def get_timestamp() -> str:
    """Return a compact ISO-8601 timestamp string."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def flatten_dict(d: dict, sep: str = "_", _prefix: str = "") -> dict:
    """Recursively flatten a nested dict, joining keys with *sep*."""
    out: dict = {}
    for key, value in d.items():
        full_key = f"{_prefix}{sep}{key}" if _prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, sep=sep, _prefix=full_key))
        else:
            out[full_key] = value
    return out


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj
