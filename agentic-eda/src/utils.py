"""Utility helpers shared across the project."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd



def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path



def reset_directory(path: Path) -> Path:
    """Ensure directory exists and remove its existing contents."""
    ensure_dir(path)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)
    return path



def sanitize_name(value: str) -> str:
    """Create filesystem-safe folder/file names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "dataset"


def build_llm_label(provider: str, model_name: str, tag: str | None = None) -> str:
    """Build a compact filesystem-safe label for an LLM configuration."""
    base = f"{provider}_{model_name}"
    if tag:
        base = f"{base}_{tag}"
    return sanitize_name(base.lower())



def to_serializable(value: Any) -> Any:
    """Convert common numpy/pandas objects into JSON-serializable primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [to_serializable(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return {str(k): to_serializable(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return [
            {str(k): to_serializable(v) for k, v in row.items()}
            for row in value.to_dict(orient="records")
        ]
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    return value



def save_json(payload: Any, path: Path) -> None:
    """Save content as pretty JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)



def save_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dictionaries to CSV (header inferred from union of keys)."""
    ensure_dir(path.parent)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: to_serializable(row.get(k)) for k in all_keys})



def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text content."""
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")



def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division helper."""
    if denominator == 0:
        return default
    return numerator / denominator
