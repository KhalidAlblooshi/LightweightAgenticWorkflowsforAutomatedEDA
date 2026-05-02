"""Matplotlib chart generation for recommended EDA visuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir



def _safe_filename(prefix: str, value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return f"{prefix}_{cleaned}.png"



def _save_current_figure(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)



def _plot_missing_values(df: pd.DataFrame, charts_dir: Path) -> str | None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar(missing.index.astype(str), missing.values)
    plt.title("Missing Values by Column")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing count")
    return _save_current_figure(charts_dir / "missing_values_bar.png")



def _plot_histogram(df: pd.DataFrame, col: str, charts_dir: Path) -> str | None:
    if col not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[col]):
        return None

    series = df[col].dropna()
    if series.empty:
        return None

    plt.figure(figsize=(7, 4))
    plt.hist(series, bins=20, edgecolor="black", alpha=0.8)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    return _save_current_figure(charts_dir / _safe_filename("hist", col))



def _plot_categorical_bar(df: pd.DataFrame, col: str, charts_dir: Path) -> str | None:
    if col not in df.columns:
        return None

    full_counts = df[col].fillna("<NA>").astype(str).value_counts(dropna=False)
    if full_counts.empty:
        return None

    unique_count = int(len(full_counts))
    singleton_ratio = float((full_counts == 1).sum() / max(unique_count, 1))
    max_unique_allowed = min(30, max(int(len(df) * 0.2), 8))
    if unique_count > max_unique_allowed:
        return None
    if singleton_ratio > 0.55 and unique_count > 12:
        return None

    value_counts = full_counts.head(15)
    plt.figure(figsize=(8, 4))
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Category Counts: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    return _save_current_figure(charts_dir / _safe_filename("catbar", col))



def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], charts_dir: Path) -> str | None:
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr(numeric_only=True)
    if corr.empty:
        return None

    plt.figure(figsize=(8, 6))
    img = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.title("Correlation Heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    return _save_current_figure(charts_dir / "correlation_heatmap.png")



def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, charts_dir: Path) -> str | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        return None

    subset = df[[x_col, y_col]].dropna()
    if subset.empty:
        return None

    plt.figure(figsize=(7, 5))
    plt.scatter(subset[x_col], subset[y_col], alpha=0.6, s=16)
    plt.title(f"Scatter: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    return _save_current_figure(charts_dir / _safe_filename("scatter", f"{x_col}_vs_{y_col}"))



def _plot_target_box(df: pd.DataFrame, target_col: str, num_col: str, charts_dir: Path) -> str | None:
    if target_col not in df.columns or num_col not in df.columns:
        return None

    groups = []
    labels = []
    for cat_value, group_df in df[[target_col, num_col]].dropna().groupby(target_col):
        groups.append(group_df[num_col].values)
        labels.append(str(cat_value))

    if len(groups) < 2:
        return None

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, labels=labels, patch_artist=True)
    plt.title(f"{num_col} by {target_col}")
    plt.xlabel(target_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha="right")
    return _save_current_figure(charts_dir / _safe_filename("target_box", f"{num_col}_by_{target_col}"))



def _plot_target_bar_mean(df: pd.DataFrame, cat_col: str, target_col: str, charts_dir: Path) -> str | None:
    if cat_col not in df.columns or target_col not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return None

    cat_counts = df[cat_col].dropna().astype(str).value_counts(dropna=False)
    if cat_counts.empty:
        return None

    unique_count = int(len(cat_counts))
    singleton_ratio = float((cat_counts == 1).sum() / max(unique_count, 1))
    if unique_count > 12:
        return None
    if singleton_ratio > 0.45 and unique_count > 8:
        return None

    grouped = (
        df[[cat_col, target_col]]
        .dropna()
        .groupby(cat_col, observed=False)[target_col]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    if grouped.empty:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.title(f"Mean {target_col} by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel(f"Mean {target_col}")
    plt.xticks(rotation=45, ha="right")
    return _save_current_figure(charts_dir / _safe_filename("target_mean", f"{target_col}_by_{cat_col}"))



def generate_charts(
    df: pd.DataFrame,
    profile: dict[str, Any],
    recommendations: list[dict[str, Any]],
    charts_dir: Path,
) -> dict[str, Any]:
    """Generate charts based on recommendations and return output metadata."""
    ensure_dir(charts_dir)

    chart_paths: list[str] = []
    failures: list[str] = []

    numeric_cols = profile.get("numeric_columns", [])
    id_like_cols = set(profile.get("id_like_columns", []))
    missing_done = False
    heatmap_done = False

    for rec in recommendations:
        chart_type = rec.get("chart_type")
        created: str | None = None

        try:
            if chart_type == "missing_bar":
                if not missing_done:
                    created = _plot_missing_values(df, charts_dir)
                    missing_done = True
            elif chart_type == "histogram":
                x_col = str(rec.get("x"))
                if x_col in id_like_cols:
                    failures.append(f"{chart_type}({x_col}): skipped identifier-like column.")
                    continue
                created = _plot_histogram(df, x_col, charts_dir)
            elif chart_type == "bar":
                x_col = str(rec.get("x"))
                if x_col in id_like_cols:
                    failures.append(f"{chart_type}({x_col}): skipped identifier-like column.")
                    continue
                created = _plot_categorical_bar(df, x_col, charts_dir)
            elif chart_type == "correlation_heatmap":
                if not heatmap_done:
                    created = _plot_correlation_heatmap(df, numeric_cols, charts_dir)
                    heatmap_done = created is not None
            elif chart_type == "scatter":
                x_col = str(rec.get("x"))
                y_col = str(rec.get("y"))
                if x_col in id_like_cols or y_col in id_like_cols:
                    failures.append(f"{chart_type}({x_col},{y_col}): skipped identifier-like column.")
                    continue
                created = _plot_scatter(df, x_col, y_col, charts_dir)
            elif chart_type == "target_box":
                x_col = str(rec.get("x"))
                y_col = str(rec.get("y"))
                if x_col in id_like_cols or y_col in id_like_cols:
                    failures.append(f"{chart_type}({x_col},{y_col}): skipped identifier-like column.")
                    continue
                created = _plot_target_box(df, x_col, y_col, charts_dir)
            elif chart_type == "target_bar_mean":
                x_col = str(rec.get("x"))
                y_col = str(rec.get("y"))
                if x_col in id_like_cols:
                    failures.append(f"{chart_type}({x_col}): skipped identifier-like column.")
                    continue
                created = _plot_target_bar_mean(df, x_col, y_col, charts_dir)
        except Exception as exc:  # pragma: no cover - defensive plotting guard
            failures.append(f"{chart_type}: {exc}")

        if created and created not in chart_paths:
            chart_paths.append(created)

    return {
        "charts_generated": len(chart_paths),
        "chart_paths": chart_paths,
        "plot_failures": failures,
    }
