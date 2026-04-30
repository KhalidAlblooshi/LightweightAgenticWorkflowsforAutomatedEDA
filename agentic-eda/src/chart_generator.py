"""Chart generator — renders and saves all visualizations using matplotlib only."""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir, sanitize_name


def generate_charts(
    df: pd.DataFrame,
    profile: dict,
    recommendations: list,
    output_dir: Path,
    dataset_name: str,
    mode: str,
) -> list:
    """Render charts from *recommendations* and save them as PNGs.

    Parameters
    ----------
    df:
        Source DataFrame.
    profile:
        Dataset profile dict.
    recommendations:
        List of recommendation dicts from the visualization recommender.
    output_dir:
        Root output directory.
    dataset_name:
        Used in filenames and titles.
    mode:
        Agent mode string (used in filenames).

    Returns
    -------
    List of saved file path strings (relative to *output_dir*).
    """
    charts_dir = output_dir / "charts" / sanitize_name(dataset_name) / mode
    ensure_dir(charts_dir)

    saved: list = []
    numeric_cols = profile["numeric_columns"]
    categorical_cols = profile["categorical_columns"]
    target_col = profile.get("likely_target_col")

    # Track which chart types have already been rendered to avoid duplicates
    rendered = set()

    for rec in recommendations:
        chart_type = rec["chart_type"]
        cols = rec.get("columns", [])

        try:
            if chart_type == "missing_values_bar" and "missing_values_bar" not in rendered:
                path = _missing_values_bar(df, profile, charts_dir, dataset_name)
                if path:
                    saved.append(str(path))
                rendered.add(chart_type)

            elif chart_type == "numeric_histogram":
                col = cols[0] if cols else None
                if col and col in df.columns:
                    key = f"hist_{col}"
                    if key not in rendered:
                        path = _histogram(df, col, charts_dir, dataset_name)
                        if path:
                            saved.append(str(path))
                        rendered.add(key)

            elif chart_type == "categorical_bar":
                col = cols[0] if cols else None
                if col and col in df.columns:
                    key = f"catbar_{col}"
                    if key not in rendered:
                        path = _categorical_bar(df, col, charts_dir, dataset_name)
                        if path:
                            saved.append(str(path))
                        rendered.add(key)

            elif chart_type == "correlation_heatmap" and "correlation_heatmap" not in rendered:
                if len(numeric_cols) >= 2:
                    path = _correlation_heatmap(df, numeric_cols, charts_dir, dataset_name)
                    if path:
                        saved.append(str(path))
                rendered.add("correlation_heatmap")

            elif chart_type == "scatter_plot" and len(cols) == 2:
                key = f"scatter_{'_'.join(cols)}"
                if key not in rendered:
                    path = _scatter_plot(df, cols[0], cols[1], charts_dir, dataset_name)
                    if path:
                        saved.append(str(path))
                    rendered.add(key)

            elif chart_type in ("target_distribution_bar", "target_histogram"):
                if target_col and target_col in df.columns:
                    key = f"target_dist_{target_col}"
                    if key not in rendered:
                        path = _target_distribution(df, target_col, charts_dir, dataset_name)
                        if path:
                            saved.append(str(path))
                        rendered.add(key)

            elif chart_type == "box_plot_by_target" and len(cols) == 2:
                feat_col, tgt_col = cols
                key = f"box_{feat_col}_{tgt_col}"
                if key not in rendered and feat_col in df.columns and tgt_col in df.columns:
                    path = _box_by_target(df, feat_col, tgt_col, charts_dir, dataset_name)
                    if path:
                        saved.append(str(path))
                    rendered.add(key)

        except Exception as exc:
            print(f"  [chart_generator] Warning: failed to render '{chart_type}': {exc}")

    return saved


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _missing_values_bar(df, profile, charts_dir, dataset_name):
    missing = {c: v for c, v in profile["missing_counts"].items() if v > 0}
    if not missing:
        return None
    cols = list(missing.keys())
    vals = [missing[c] for c in cols]
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.7), 4))
    ax.bar(range(len(cols)), vals, color="#d62728")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Missing count")
    ax.set_title(f"Missing Values — {dataset_name}")
    path = charts_dir / "missing_values_bar.png"
    return _save(fig, path)


def _histogram(df, col, charts_dir, dataset_name):
    data = df[col].dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color="#1f77b4", edgecolor="white")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {col} — {dataset_name}")
    path = charts_dir / f"hist_{sanitize_name(col)}.png"
    return _save(fig, path)


def _categorical_bar(df, col, charts_dir, dataset_name):
    counts = df[col].value_counts().head(15)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.8), 4))
    ax.bar(range(len(counts)), counts.values, color="#2ca02c")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"Category Counts: {col} — {dataset_name}")
    path = charts_dir / f"catbar_{sanitize_name(col)}.png"
    return _save(fig, path)


def _correlation_heatmap(df, numeric_cols, charts_dir, dataset_name):
    corr = df[numeric_cols].corr()
    n = len(numeric_cols)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c if len(c) <= 12 else c[:10] + ".." for c in numeric_cols]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Correlation Heatmap — {dataset_name}")
    path = charts_dir / "correlation_heatmap.png"
    return _save(fig, path)


def _scatter_plot(df, col_x, col_y, charts_dir, dataset_name):
    data = df[[col_x, col_y]].dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[col_x], data[col_y], alpha=0.4, s=15, color="#9467bd")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"{col_x} vs {col_y} — {dataset_name}")
    path = charts_dir / f"scatter_{sanitize_name(col_x)}_{sanitize_name(col_y)}.png"
    return _save(fig, path)


def _target_distribution(df, target_col, charts_dir, dataset_name):
    if df[target_col].dtype == object or str(df[target_col].dtype) == "category":
        counts = df[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(max(5, len(counts) * 0.8), 4))
        ax.bar(range(len(counts)), counts.values, color="#ff7f0e")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution: {target_col} — {dataset_name}")
    else:
        data = df[target_col].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=20, color="#ff7f0e", edgecolor="white")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Target Distribution: {target_col} — {dataset_name}")
    path = charts_dir / f"target_dist_{sanitize_name(target_col)}.png"
    return _save(fig, path)


def _box_by_target(df, feat_col, target_col, charts_dir, dataset_name):
    groups = df.groupby(target_col)[feat_col].apply(lambda x: x.dropna().tolist())
    if groups.empty:
        return None
    labels = [str(k) for k in groups.index]
    data = [groups[k] for k in groups.index]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), 4))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_xlabel(target_col)
    ax.set_ylabel(feat_col)
    ax.set_title(f"{feat_col} by {target_col} — {dataset_name}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    path = charts_dir / f"box_{sanitize_name(feat_col)}_by_{sanitize_name(target_col)}.png"
    return _save(fig, path)
