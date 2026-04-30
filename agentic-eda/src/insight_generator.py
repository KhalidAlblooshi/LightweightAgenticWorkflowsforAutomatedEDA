"""Insight generator — produces natural-language observations from EDA results."""

import pandas as pd


def generate_insights(df: pd.DataFrame, profile: dict, tool_results: dict) -> list:
    """Produce 10-20 insight strings from *tool_results* and *profile*.

    Parameters
    ----------
    df:
        The source DataFrame.
    profile:
        Dataset profile dict.
    tool_results:
        Dict mapping tool names to their result dicts.

    Returns
    -------
    List of non-empty insight strings.
    """
    insights: list = []
    name = profile["dataset_name"]
    n_rows = profile["n_rows"]
    n_cols = profile["n_cols"]

    # --- Dataset size ---
    insights.append(
        f"The dataset '{name}' contains {n_rows:,} rows and {n_cols} columns."
    )
    if n_rows < 100:
        insights.append(
            "The dataset is very small (<100 rows); statistical conclusions may be unreliable."
        )
    elif n_rows > 50_000:
        insights.append(
            "The dataset is large (>50 000 rows), which supports robust statistical analyses."
        )

    # --- Missing values ---
    mv = tool_results.get("missing_value_analysis", {})
    total_missing = mv.get("total_missing_cells", 0)
    high_missing_cols = mv.get("columns_above_20pct_missing", [])
    if total_missing == 0:
        insights.append("No missing values were detected; the dataset appears complete.")
    else:
        pct = mv.get("overall_missing_pct", 0)
        insights.append(
            f"{total_missing:,} missing cells detected ({pct:.1f}% of all values)."
        )
        if high_missing_cols:
            cols_str = ", ".join(high_missing_cols)
            insights.append(
                f"Columns with >20% missing data: {cols_str}. "
                "Consider imputation or removal before modelling."
            )

    # --- Duplicates ---
    dup = tool_results.get("duplicate_analysis", {})
    dup_count = dup.get("duplicate_row_count", 0)
    if dup_count > 0:
        dup_pct = dup.get("duplicate_row_pct", 0)
        insights.append(
            f"{dup_count} duplicate rows detected ({dup_pct:.1f}%). "
            "Remove duplicates before training ML models."
        )
    else:
        insights.append("No duplicate rows were found.")

    # --- Numeric distributions ---
    num_sum = tool_results.get("numeric_summary", {})
    skewed_cols = []
    for col, stats in num_sum.items():
        if isinstance(stats, dict):
            skew = stats.get("skewness")
            if skew is not None and abs(skew) > 1.0:
                skewed_cols.append((col, skew))
    if skewed_cols:
        examples = ", ".join(
            f"'{c}' (skew={s:.2f})" for c, s in skewed_cols[:4]
        )
        insights.append(
            f"Highly skewed numeric columns (|skew|>1): {examples}. "
            "Log or power transforms may improve model performance."
        )

    # --- Correlations ---
    corr_res = tool_results.get("correlation_analysis", {})
    top_pairs = corr_res.get("top_correlated_pairs", [])
    if top_pairs:
        pair = top_pairs[0]
        insights.append(
            f"Strongest correlation: '{pair.get('col1')}' ↔ '{pair.get('col2')}' "
            f"(r={pair.get('correlation', 0):.3f}). "
            "Multicollinearity may affect linear models."
        )
    if len(top_pairs) > 3:
        insights.append(
            f"{len(top_pairs)} feature pairs have |r|>0.3, suggesting potential redundancy."
        )

    # --- Outliers ---
    outlier_res = tool_results.get("outlier_detection", {})
    high_outlier_cols = [
        col for col, stats in outlier_res.items()
        if isinstance(stats, dict) and stats.get("outlier_pct", 0) > 5
    ]
    if high_outlier_cols:
        cols_str = ", ".join(high_outlier_cols[:4])
        insights.append(
            f"Columns with >5% outliers (IQR method): {cols_str}. "
            "Investigate these rows and consider robust scalers."
        )
    else:
        numeric_cols = profile["numeric_columns"]
        if numeric_cols:
            insights.append("No numeric column has more than 5% IQR-flagged outliers.")

    # --- Categorical columns ---
    cat_sum = tool_results.get("categorical_summary", {})
    high_cardinality = []
    for col, stats in cat_sum.items():
        if isinstance(stats, dict):
            unique = stats.get("unique_count", 0)
            if unique > 50:
                high_cardinality.append(col)
    if high_cardinality:
        cols_str = ", ".join(high_cardinality)
        insights.append(
            f"High-cardinality categorical columns (>50 unique values): {cols_str}. "
            "Encoding strategy should be chosen carefully."
        )

    # --- Target variable ---
    tgt = profile.get("likely_target_col")
    ta = tool_results.get("target_aware_analysis", {})
    if tgt and not ta.get("skipped"):
        target_type = ta.get("target_type", "unknown")
        if target_type == "categorical":
            class_dist = ta.get("class_distribution", {})
            if class_dist:
                vals = list(class_dist.values())
                if vals:
                    max_pct = max(vals)
                    if max_pct > 70:
                        insights.append(
                            f"Target '{tgt}' is imbalanced — dominant class covers "
                            f"{max_pct:.1f}% of rows. Consider resampling or class weights."
                        )
                    else:
                        insights.append(
                            f"Target '{tgt}' is relatively balanced across classes."
                        )
        elif target_type == "numeric":
            insights.append(
                f"Target '{tgt}' is numeric; regression models are appropriate."
            )

    # --- Visualization ---
    viz_recs = tool_results.get("visualization_recommendation", {})
    n_recs = len(viz_recs) if isinstance(viz_recs, list) else 0
    if n_recs > 0:
        insights.append(
            f"{n_recs} visualizations were recommended to aid exploratory analysis."
        )

    # --- General data quality note ---
    issues = []
    if total_missing > 0:
        issues.append("missing values")
    if dup_count > 0:
        issues.append("duplicate rows")
    if high_outlier_cols:
        issues.append("outliers")
    if issues:
        insights.append(
            f"Data quality issues found: {', '.join(issues)}. "
            "Address these before downstream modelling."
        )
    else:
        insights.append(
            "No major data quality issues (missing values, duplicates, or heavy outliers) found."
        )

    return insights
