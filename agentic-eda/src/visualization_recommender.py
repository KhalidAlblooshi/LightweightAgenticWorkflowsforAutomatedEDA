"""Recommend visualization types based on dataset/profile context."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _column_stats(profile: dict[str, Any], col: str) -> dict[str, Any]:
    return profile.get("column_stats", {}).get(col, {})


def _is_identifier_like(profile: dict[str, Any], col: str) -> bool:
    if col in set(profile.get("id_like_columns", [])):
        return True
    stats = _column_stats(profile, col)
    return bool(stats.get("is_identifier_like", False))


def _categorical_eligible(
    profile: dict[str, Any],
    col: str,
    row_count: int,
    max_unique: int = 30,
    max_singleton_ratio: float = 0.55,
) -> bool:
    if _is_identifier_like(profile, col):
        return False

    stats = _column_stats(profile, col)
    unique_count = int(stats.get("unique_count", 0))
    unique_ratio = float(stats.get("unique_ratio", 0.0))
    singleton_ratio = float(stats.get("singleton_ratio", 0.0))

    if unique_count < 2:
        return False
    if unique_count > min(max_unique, max(int(row_count * 0.2), 8)):
        return False
    if unique_ratio > 0.4:
        return False
    if singleton_ratio > max_singleton_ratio and unique_count > 12:
        return False
    return True


def _numeric_hist_eligible(
    profile: dict[str, Any],
    col: str,
    row_count: int,
) -> bool:
    if _is_identifier_like(profile, col):
        return False

    stats = _column_stats(profile, col)
    unique_count = int(stats.get("unique_count", 0))
    unique_ratio = float(stats.get("unique_ratio", 0.0))
    non_null_count = int(stats.get("non_null_count", 0))

    if non_null_count < max(20, int(row_count * 0.02)):
        return False
    if unique_count < 3:
        return False
    if unique_ratio > 0.995 and unique_count > 60:
        return False
    return True


def recommend_visualizations(
    df: pd.DataFrame,
    profile: dict[str, Any],
    analysis_results: dict[str, Any],
    guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate deterministic chart recommendations for downstream plotting."""
    recommendations: list[dict[str, Any]] = []
    numeric_cols: list[str] = profile.get("numeric_columns", [])
    categorical_cols: list[str] = profile.get("categorical_columns", [])
    target_col: str | None = profile.get("likely_target_col")
    row_count = int(profile.get("row_count", len(df)))

    guidance = guidance or {}
    focus_cols = set(guidance.get("focus_columns", []) or [])
    avoid_cols = set(guidance.get("avoid_columns", []) or [])

    excluded_identifier_like: list[str] = []
    excluded_high_cardinality_categorical: list[str] = []
    excluded_user_avoid: list[str] = []

    numeric_candidates: list[str] = []
    for col in numeric_cols:
        if col in avoid_cols:
            excluded_user_avoid.append(col)
            continue
        if _is_identifier_like(profile, col):
            excluded_identifier_like.append(col)
            continue
        if _numeric_hist_eligible(profile, col, row_count):
            numeric_candidates.append(col)

    categorical_candidates: list[str] = []
    for col in categorical_cols:
        if col in avoid_cols:
            excluded_user_avoid.append(col)
            continue
        if _is_identifier_like(profile, col):
            excluded_identifier_like.append(col)
            continue
        if _categorical_eligible(profile, col, row_count):
            categorical_candidates.append(col)
        else:
            excluded_high_cardinality_categorical.append(col)

    def _priority(col: str) -> tuple[int, float, int, str]:
        stats = _column_stats(profile, col)
        focus_rank = 0 if col in focus_cols else 1
        unique_ratio = float(stats.get("unique_ratio", 0.0))
        unique_count = int(stats.get("unique_count", 0))
        return (focus_rank, unique_ratio, unique_count, str(col))

    numeric_candidates = sorted(numeric_candidates, key=_priority)
    categorical_candidates = sorted(categorical_candidates, key=_priority)

    missing_info = analysis_results.get("missing_value_analysis", {})
    missing_cols = missing_info.get("columns_with_missing", [])
    if missing_cols:
        recommendations.append(
            {
                "chart_type": "missing_bar",
                "x": "column",
                "y": "missing_count",
                "title": "Missing values by column",
                "reason": "Columns with missing values were detected.",
                "priority": 1,
            }
        )

    for col in numeric_candidates[:6]:
        recommendations.append(
            {
                "chart_type": "histogram",
                "x": col,
                "y": None,
                "title": f"Distribution of {col}",
                "reason": "Numeric distribution overview.",
                "priority": 2,
            }
        )

    for col in categorical_candidates[:4]:
        recommendations.append(
            {
                "chart_type": "bar",
                "x": col,
                "y": "count",
                "title": f"Category frequencies for {col}",
                "reason": "Categorical frequency comparison.",
                "priority": 2,
            }
        )

    heatmap_numeric_cols = [
        col
        for col in numeric_cols
        if col not in avoid_cols and not _is_identifier_like(profile, col)
    ]
    if len(heatmap_numeric_cols) >= 2:
        recommendations.append(
            {
                "chart_type": "correlation_heatmap",
                "x": "numeric_features",
                "y": "numeric_features",
                "title": "Correlation heatmap",
                "reason": "Multiple numeric columns are available.",
                "priority": 1,
            }
        )

    strongest_pair = analysis_results.get("correlation_analysis", {}).get("strongest_pair")
    if strongest_pair and strongest_pair.get("feature_1") and strongest_pair.get("feature_2"):
        x_col = strongest_pair["feature_1"]
        y_col = strongest_pair["feature_2"]
        if (
            x_col not in avoid_cols
            and y_col not in avoid_cols
            and not _is_identifier_like(profile, x_col)
            and not _is_identifier_like(profile, y_col)
        ):
            recommendations.append(
                {
                    "chart_type": "scatter",
                    "x": x_col,
                    "y": y_col,
                    "title": "Strongest numeric correlation",
                    "reason": "Useful for validating linear trend strength.",
                    "priority": 1,
                }
            )

    if target_col and target_col in df.columns:
        target_unique = df[target_col].nunique(dropna=True)
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

        if target_is_numeric and target_unique > 15:
            target_cat_candidates = [
                col
                for col in categorical_candidates
                if _categorical_eligible(profile, col, row_count, max_unique=12, max_singleton_ratio=0.45)
            ]
            for cat_col in target_cat_candidates[:2]:
                recommendations.append(
                    {
                        "chart_type": "target_bar_mean",
                        "x": cat_col,
                        "y": target_col,
                        "title": f"Mean {target_col} by {cat_col}",
                        "reason": "Target-aware comparison across categories.",
                        "priority": 3,
                    }
                )
        elif (not target_is_numeric) or target_unique <= 12:
            target_numeric_candidates = [c for c in numeric_candidates if c != target_col]
            for num_col in target_numeric_candidates[:2]:
                recommendations.append(
                    {
                        "chart_type": "target_box",
                        "x": target_col,
                        "y": num_col,
                        "title": f"{num_col} by target ({target_col})",
                        "reason": "Highlights class-wise numeric spread.",
                        "priority": 3,
                    }
                )

    # deterministic ordering
    recommendations.sort(key=lambda rec: (rec["priority"], rec["chart_type"], str(rec.get("x"))))

    return {
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "filter_diagnostics": {
            "excluded_identifier_like_columns": sorted(set(excluded_identifier_like)),
            "excluded_high_cardinality_categorical_columns": sorted(set(excluded_high_cardinality_categorical)),
            "excluded_user_avoid_columns": sorted(set(excluded_user_avoid)),
            "numeric_candidates_used": numeric_candidates,
            "categorical_candidates_used": categorical_candidates,
        },
    }
