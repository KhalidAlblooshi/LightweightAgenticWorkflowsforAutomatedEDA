"""Dataset profiling utilities used by all agents."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.utils import to_serializable

TARGET_HINTS = [
    "target",
    "label",
    "class",
    "outcome",
    "result",
    "survived",
    "quality",
    "income",
    "price",
    "score",
    "grade",
    "churn",
    "default",
]

IDENTIFIER_NAME_HINTS = {
    "id",
    "uuid",
    "guid",
    "identifier",
    "index",
    "serial",
    "record",
}


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _tokenize_column_name(name: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", str(name).lower())
    return [token for token in tokens if token]


def _target_hint_rank(column_name: str) -> int | None:
    normalized = str(column_name).lower()
    tokens = set(_tokenize_column_name(column_name))
    for idx, hint in enumerate(TARGET_HINTS):
        if normalized == hint or hint in tokens:
            return idx
    return None


def _is_monotonic_integer_like(series: pd.Series) -> bool:
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return False

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return False

    rounded = np.round(values.values)
    if not np.allclose(values.values, rounded, atol=1e-9):
        return False

    unique_vals = np.unique(rounded)
    if len(unique_vals) < 10:
        return False
    if len(unique_vals) != len(values):
        return False

    sorted_vals = np.sort(unique_vals)
    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return False
    return float(np.mean(diffs == 1)) >= 0.95


def _is_identifier_like_column(df: pd.DataFrame, column_name: str) -> bool:
    if column_name not in df.columns:
        return False

    series = df[column_name].dropna()
    if series.empty:
        return False

    non_null_count = len(series)
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = _safe_div(unique_count, non_null_count)
    tokens = set(_tokenize_column_name(column_name))

    if tokens.intersection(IDENTIFIER_NAME_HINTS) and unique_ratio >= 0.5:
        return True

    if unique_count >= 50 and unique_ratio >= 0.98:
        return True

    if _is_monotonic_integer_like(series):
        return True

    return False


def _build_column_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = _safe_div(unique_count, non_null_count)

        value_counts = non_null.astype(str).value_counts(dropna=False)
        singleton_count = int((value_counts == 1).sum()) if not value_counts.empty else 0
        singleton_ratio = _safe_div(singleton_count, len(value_counts))

        stats[col] = {
            "dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float((series.isna().mean() * 100.0)),
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "singleton_ratio": singleton_ratio,
            "is_identifier_like": _is_identifier_like_column(df, col),
        }
    return stats


def _likely_target_column(df: pd.DataFrame, column_stats: dict[str, dict[str, Any]]) -> str | None:
    """Heuristic target detection that avoids obvious identifier columns."""
    candidate_rows: list[tuple[float, str]] = []

    for idx, col in enumerate(df.columns):
        stats = column_stats.get(col, {})
        unique_count = int(stats.get("unique_count", 0))
        unique_ratio = float(stats.get("unique_ratio", 0.0))
        is_identifier_like = bool(stats.get("is_identifier_like", False))
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        hint_rank = _target_hint_rank(col)

        score = 0.0
        if hint_rank is not None:
            score += 120.0 - float(hint_rank * 5)

        if is_identifier_like:
            score -= 120.0

        if 2 <= unique_count <= 20:
            score += 28.0
        elif 21 <= unique_count <= 60:
            score += 12.0
        elif unique_ratio > 0.85:
            score -= 22.0

        if is_numeric and unique_count > 120:
            score -= 8.0
        if not is_numeric and unique_count > 40:
            score -= 20.0

        if idx == len(df.columns) - 1:
            score += 6.0

        candidate_rows.append((score, col))

    candidate_rows.sort(key=lambda row: row[0], reverse=True)
    if not candidate_rows:
        return None

    best_score, best_col = candidate_rows[0]
    if best_score < 8.0:
        return None
    return best_col


def profile_dataset(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Create a compact, LLM-safe profile for analysis and planning."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in df.columns if c not in numeric_columns]

    missing_counts = df.isna().sum().to_dict()
    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    column_stats = _build_column_stats(df)

    numeric_column_stats = {c: column_stats.get(c, {}) for c in numeric_columns}
    categorical_column_stats = {c: column_stats.get(c, {}) for c in categorical_columns}
    id_like_columns = [c for c, stats in column_stats.items() if bool(stats.get("is_identifier_like"))]

    profile = {
        "dataset_name": dataset_name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "duplicate_rows": int(df.duplicated().sum()),
        "column_stats": column_stats,
        "numeric_column_stats": numeric_column_stats,
        "categorical_column_stats": categorical_column_stats,
        "id_like_columns": id_like_columns,
        "likely_target_col": _likely_target_column(df, column_stats),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    return to_serializable(profile)
