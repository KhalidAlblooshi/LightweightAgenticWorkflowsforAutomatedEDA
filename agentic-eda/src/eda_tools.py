"""EDA tools — each tool takes (df, profile, **kwargs) and returns a JSON-serializable dict."""

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils import convert_numpy_types
from src.visualization_recommender import recommend_visualizations


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def dataset_overview(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """High-level overview of the dataset."""
    result = {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "column_names": list(df.columns),
        "dtype_summary": {
            "numeric": len(profile["numeric_columns"]),
            "categorical": len(profile["categorical_columns"]),
            "datetime": len(profile["datetime_columns"]),
        },
        "memory_usage_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 2),
        "duplicate_row_count": int(df.duplicated().sum()),
        "total_cells": int(df.shape[0] * df.shape[1]),
    }
    return convert_numpy_types(result)


def missing_value_analysis(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Per-column and aggregate missing value statistics."""
    per_col = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        pct = round(float(cnt / len(df) * 100), 2) if len(df) > 0 else 0.0
        per_col[col] = {"missing_count": cnt, "missing_pct": pct}

    total_cells = int(df.shape[0] * df.shape[1])
    total_missing = int(df.isna().sum().sum())
    overall_pct = round(float(total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    cols_above_20 = [
        col for col, s in per_col.items() if s["missing_pct"] > 20
    ]

    return convert_numpy_types({
        "per_column": per_col,
        "total_missing_cells": total_missing,
        "total_cells": total_cells,
        "overall_missing_pct": overall_pct,
        "columns_above_20pct_missing": cols_above_20,
    })


def duplicate_analysis(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Count and percentage of exact duplicate rows."""
    dup_count = int(df.duplicated().sum())
    dup_pct = round(float(dup_count / len(df) * 100), 2) if len(df) > 0 else 0.0
    return convert_numpy_types({
        "duplicate_row_count": dup_count,
        "duplicate_row_pct": dup_pct,
        "total_rows": int(len(df)),
    })


def numeric_summary(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Descriptive statistics for every numeric column."""
    result = {}
    for col in profile["numeric_columns"]:
        series = df[col].dropna()
        if series.empty:
            result[col] = {"error": "all values missing"}
            continue
        skewness = float(series.skew()) if len(series) > 2 else None
        kurt = float(series.kurtosis()) if len(series) > 3 else None
        result[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 6),
            "median": round(float(series.median()), 6),
            "std": round(float(series.std()), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "q25": round(float(series.quantile(0.25)), 6),
            "q75": round(float(series.quantile(0.75)), 6),
            "skewness": round(skewness, 6) if skewness is not None else None,
            "kurtosis": round(kurt, 6) if kurt is not None else None,
        }
    return convert_numpy_types(result)


def categorical_summary(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Value counts, unique count, and mode for categorical columns."""
    result = {}
    for col in profile["categorical_columns"]:
        series = df[col].dropna()
        if series.empty:
            result[col] = {"error": "all values missing"}
            continue
        vc = series.value_counts()
        top10 = {str(k): int(v) for k, v in vc.head(10).items()}
        mode_val = str(series.mode().iloc[0]) if not series.mode().empty else None
        result[col] = {
            "unique_count": int(series.nunique()),
            "top_value_counts": top10,
            "mode": mode_val,
            "total_non_null": int(len(series)),
        }
    return convert_numpy_types(result)


def correlation_analysis(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Pearson correlation matrix and top strongly correlated pairs."""
    num_cols = profile["numeric_columns"]
    if len(num_cols) < 2:
        return {"skipped": True, "reason": "Fewer than 2 numeric columns."}

    corr_df = df[num_cols].corr(method="pearson")
    corr_dict = {
        col: {c: (round(float(v), 6) if not math.isnan(v) else None)
              for c, v in corr_df[col].items()}
        for col in corr_df.columns
    }

    # Find top pairs (upper triangle, |r| > 0.3)
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if not math.isnan(val) and abs(val) > 0.3:
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(float(val), 6),
                })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return convert_numpy_types({
        "correlation_matrix": corr_dict,
        "top_correlated_pairs": pairs[:5],
    })


def outlier_detection(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """IQR-based outlier detection for each numeric column."""
    result = {}
    for col in profile["numeric_columns"]:
        series = df[col].dropna()
        if len(series) < 4:
            result[col] = {"error": "not enough data"}
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        pct = round(float(count / len(series) * 100), 2)
        result[col] = {
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(iqr, 6),
            "lower_fence": round(lower, 6),
            "upper_fence": round(upper, 6),
            "outlier_count": count,
            "outlier_pct": pct,
        }
    return convert_numpy_types(result)


def target_aware_analysis(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Analysis conditioned on the likely target column."""
    target_col = profile.get("likely_target_col")
    if not target_col or target_col not in df.columns:
        return {"skipped": True, "reason": "No likely target column identified."}

    num_cols = [c for c in profile["numeric_columns"] if c != target_col]
    target_series = df[target_col].dropna()

    if target_col in profile["categorical_columns"] or target_series.dtype == object:
        class_dist = {
            str(k): round(float(v / len(target_series) * 100), 2)
            for k, v in target_series.value_counts().items()
        }
        group_means = {}
        for col in num_cols[:10]:
            gm = df.groupby(target_col)[col].mean()
            group_means[col] = {str(k): round(float(v), 4) for k, v in gm.items()
                                if not math.isnan(v)}
        return convert_numpy_types({
            "target_col": target_col,
            "target_type": "categorical",
            "class_distribution": class_dist,
            "group_means_of_numeric_cols": group_means,
        })
    else:
        correlations = {}
        for col in num_cols[:20]:
            try:
                valid = df[[col, target_col]].dropna()
                if len(valid) > 2:
                    r, _ = scipy_stats.pearsonr(valid[col], valid[target_col])
                    correlations[col] = round(float(r), 6)
            except Exception:
                pass
        return convert_numpy_types({
            "target_col": target_col,
            "target_type": "numeric",
            "pearson_correlations_with_target": correlations,
        })


def visualization_recommendation(df: pd.DataFrame, profile: dict, **kwargs) -> list:
    """Return chart recommendations from the visualization recommender."""
    return recommend_visualizations(df, profile)


def chart_generation(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Generate and save charts; return list of saved file paths."""
    from src.chart_generator import generate_charts

    output_dir = kwargs.get("output_dir")
    mode = kwargs.get("mode", "unknown")
    recommendations = kwargs.get("recommendations", [])

    if output_dir is None:
        return {"error": "output_dir not provided via kwargs"}

    # If no recommendations passed, generate them
    if not recommendations:
        recommendations = recommend_visualizations(df, profile)

    saved_paths = generate_charts(
        df=df,
        profile=profile,
        recommendations=recommendations,
        output_dir=output_dir,
        dataset_name=profile["dataset_name"],
        mode=mode,
    )
    return {"chart_paths": saved_paths, "chart_count": len(saved_paths)}


def insight_generation(df: pd.DataFrame, profile: dict, **kwargs) -> dict:
    """Generate natural-language insights from prior tool results."""
    from src.insight_generator import generate_insights

    tool_results = kwargs.get("tool_results", {})
    insights = generate_insights(df, profile, tool_results)
    return {"insights": insights, "insight_count": len(insights)}
