"""Safe EDA tool implementations used by all workflow strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.chart_generator import generate_charts
from src.insight_generator import generate_insights
from src.visualization_recommender import recommend_visualizations


class EDATools:
    """Collection of deterministic, JSON-serializable EDA tools."""

    def __init__(self, df: pd.DataFrame, profile: dict[str, Any], run_dir: Path) -> None:
        self.df = df
        self.profile = profile
        self.run_dir = run_dir

    def _id_like_columns(self) -> set[str]:
        return set(self.profile.get("id_like_columns", []))

    # ------------------------------
    # Core analysis tools
    # ------------------------------
    def tool_dataset_overview(self, state: dict[str, Any]) -> dict[str, Any]:
        df = self.df
        return {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "numeric_column_count": len(self.profile.get("numeric_columns", [])),
            "categorical_column_count": len(self.profile.get("categorical_columns", [])),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "column_types": self.profile.get("column_types", {}),
            "sample_rows": df.head(5).to_dict(orient="records"),
        }

    def tool_missing_value_analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        df = self.df
        missing_counts = df.isna().sum().sort_values(ascending=False)
        total_missing = int(missing_counts.sum())
        total_cells = int(df.shape[0] * df.shape[1])

        columns_with_missing = [col for col, count in missing_counts.items() if count > 0]
        top_missing_columns = [
            {
                "column": str(col),
                "missing_count": int(count),
                "missing_percentage": float((count / max(len(df), 1)) * 100),
            }
            for col, count in missing_counts.items()
            if count > 0
        ]

        return {
            "total_missing_cells": total_missing,
            "missing_cell_percentage": float((total_missing / max(total_cells, 1)) * 100),
            "columns_with_missing": columns_with_missing,
            "top_missing_columns": top_missing_columns,
        }

    def tool_duplicate_analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        duplicate_mask = self.df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        return {
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": float((duplicate_count / max(len(self.df), 1)) * 100),
            "duplicate_row_indices_sample": self.df.index[duplicate_mask].tolist()[:20],
        }

    def tool_numeric_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        id_like_cols = self._id_like_columns()
        numeric_cols = [c for c in self.profile.get("numeric_columns", []) if c not in id_like_cols]
        summaries: list[dict[str, Any]] = []

        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue

            values = series.astype(float).values
            skewness = float(stats.skew(values)) if len(values) > 2 else 0.0
            kurt = float(stats.kurtosis(values)) if len(values) > 3 else 0.0

            summaries.append(
                {
                    "column": col,
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "q1": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q3": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "skewness": skewness,
                    "kurtosis": kurt,
                }
            )

        return {
            "numeric_column_count": len(numeric_cols),
            "excluded_identifier_like_columns": sorted(id_like_cols.intersection(set(self.profile.get("numeric_columns", [])))),
            "column_summaries": summaries,
        }

    def tool_categorical_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        id_like_cols = self._id_like_columns()
        categorical_cols = [c for c in self.profile.get("categorical_columns", []) if c not in id_like_cols]
        summaries: list[dict[str, Any]] = []

        for col in categorical_cols:
            series = self.df[col].astype(str).fillna("<NA>")
            counts = series.value_counts(dropna=False)
            top_categories = [
                {
                    "category": str(cat),
                    "count": int(cnt),
                    "percentage": float((cnt / max(len(series), 1)) * 100),
                }
                for cat, cnt in counts.head(10).items()
            ]

            summaries.append(
                {
                    "column": col,
                    "unique_count": int(series.nunique(dropna=False)),
                    "top_categories": top_categories,
                }
            )

        return {
            "categorical_column_count": len(categorical_cols),
            "excluded_identifier_like_columns": sorted(id_like_cols.intersection(set(self.profile.get("categorical_columns", [])))),
            "column_summaries": summaries,
        }

    def tool_correlation_analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        id_like_cols = self._id_like_columns()
        numeric_cols = [c for c in self.profile.get("numeric_columns", []) if c not in id_like_cols]
        if len(numeric_cols) < 2:
            return {
                "correlation_matrix": {},
                "strongest_pair": None,
                "top_pairs": [],
                "excluded_identifier_like_columns": sorted(id_like_cols.intersection(set(self.profile.get("numeric_columns", [])))),
            }

        corr = self.df[numeric_cols].corr(numeric_only=True)
        pairs: list[dict[str, Any]] = []

        for i, col_i in enumerate(corr.columns):
            for j, col_j in enumerate(corr.columns):
                if j <= i:
                    continue
                value = corr.iloc[i, j]
                if pd.isna(value):
                    continue
                pairs.append(
                    {
                        "feature_1": str(col_i),
                        "feature_2": str(col_j),
                        "correlation": float(value),
                        "abs_correlation": float(abs(value)),
                    }
                )

        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return {
            "correlation_matrix": corr.round(4).to_dict(),
            "strongest_pair": pairs[0] if pairs else None,
            "top_pairs": pairs[:10],
            "excluded_identifier_like_columns": sorted(id_like_cols.intersection(set(self.profile.get("numeric_columns", [])))),
        }

    def tool_outlier_detection(self, state: dict[str, Any]) -> dict[str, Any]:
        id_like_cols = self._id_like_columns()
        numeric_cols = [c for c in self.profile.get("numeric_columns", []) if c not in id_like_cols]
        outlier_rows: list[dict[str, Any]] = []

        for col in numeric_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                lower, upper = q1, q3
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

            outlier_mask = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())
            outlier_rows.append(
                {
                    "column": col,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "outlier_count": outlier_count,
                    "outlier_percentage": float((outlier_count / max(len(series), 1)) * 100),
                }
            )

        outlier_rows.sort(key=lambda x: x["outlier_count"], reverse=True)
        return {
            "ranked_outlier_columns": outlier_rows,
            "columns_with_outliers": [r["column"] for r in outlier_rows if r["outlier_count"] > 0],
            "excluded_identifier_like_columns": sorted(id_like_cols.intersection(set(self.profile.get("numeric_columns", [])))),
        }

    def tool_target_aware_analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        target_col = self.profile.get("likely_target_col")
        if not target_col or target_col not in self.df.columns:
            return {
                "target_column": None,
                "analysis_type": "none",
                "findings": [],
            }

        findings: list[str] = []
        target_series = self.df[target_col]
        id_like_cols = self._id_like_columns()
        numeric_cols = [c for c in self.profile.get("numeric_columns", []) if c != target_col and c not in id_like_cols]
        categorical_cols = [c for c in self.profile.get("categorical_columns", []) if c != target_col]

        target_is_numeric = pd.api.types.is_numeric_dtype(target_series)
        target_unique = target_series.nunique(dropna=True)

        if target_is_numeric and target_unique > 15:
            for col in numeric_cols[:6]:
                subset = self.df[[target_col, col]].dropna()
                if subset.empty:
                    continue
                corr = subset[target_col].corr(subset[col])
                if pd.notna(corr):
                    findings.append(f"Target '{target_col}' correlation with '{col}' is {corr:.3f}.")
            analysis_type = "numeric_target"
        else:
            grouped = self.df.groupby(target_col, observed=False)
            class_counts = grouped.size().sort_values(ascending=False)
            findings.append(
                "Target class distribution: "
                + ", ".join(f"{k}={v}" for k, v in class_counts.head(8).items())
                + "."
            )

            for col in numeric_cols[:6]:
                sub = self.df[[target_col, col]].dropna()
                if sub[target_col].nunique() < 2:
                    continue
                group_values = [grp[col].values for _, grp in sub.groupby(target_col, observed=False)]
                if len(group_values) >= 2 and all(len(v) > 1 for v in group_values):
                    try:
                        anova = stats.f_oneway(*group_values)
                        findings.append(
                            f"ANOVA for '{col}' across target '{target_col}' groups: p-value={anova.pvalue:.4g}."
                        )
                    except Exception:
                        continue
            analysis_type = "categorical_target"

        return {
            "target_column": target_col,
            "analysis_type": analysis_type,
            "findings": findings,
            "numeric_features_considered": numeric_cols,
            "categorical_features_considered": categorical_cols,
        }

    # ------------------------------
    # Downstream synthesis tools
    # ------------------------------
    def tool_visualization_recommendation(self, state: dict[str, Any]) -> dict[str, Any]:
        return recommend_visualizations(
            self.df,
            self.profile,
            state.get("tool_results", {}),
            guidance=state.get("llm_column_guidance"),
        )

    def tool_chart_generation(self, state: dict[str, Any]) -> dict[str, Any]:
        rec_result = state.get("tool_results", {}).get("visualization_recommendation", {})
        recommendations = rec_result.get("recommendations", [])

        charts_dir = self.run_dir / "charts"
        chart_output = generate_charts(self.df, self.profile, recommendations, charts_dir)

        # store chart paths relative to mode output directory for cleaner reports
        rel_paths = []
        for path_str in chart_output.get("chart_paths", []):
            path = Path(path_str)
            try:
                rel_paths.append(str(path.relative_to(self.run_dir)))
            except ValueError:
                rel_paths.append(str(path))
        chart_output["chart_paths"] = rel_paths
        return chart_output

    def tool_insight_generation(self, state: dict[str, Any]) -> dict[str, Any]:
        return generate_insights(state.get("tool_results", {}))



def build_tool_map(tool_impl: EDATools) -> dict[str, Any]:
    """Create dictionary mapping safe tool names to concrete callables."""
    return {
        "dataset_overview": tool_impl.tool_dataset_overview,
        "missing_value_analysis": tool_impl.tool_missing_value_analysis,
        "duplicate_analysis": tool_impl.tool_duplicate_analysis,
        "numeric_summary": tool_impl.tool_numeric_summary,
        "categorical_summary": tool_impl.tool_categorical_summary,
        "correlation_analysis": tool_impl.tool_correlation_analysis,
        "outlier_detection": tool_impl.tool_outlier_detection,
        "target_aware_analysis": tool_impl.tool_target_aware_analysis,
        "visualization_recommendation": tool_impl.tool_visualization_recommendation,
        "chart_generation": tool_impl.tool_chart_generation,
        "insight_generation": tool_impl.tool_insight_generation,
    }
