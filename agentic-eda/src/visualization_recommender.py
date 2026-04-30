"""Visualization recommender — rule-based chart suggestions."""

import pandas as pd


def recommend_visualizations(df: pd.DataFrame, profile: dict) -> list:
    """Generate chart recommendations based on dataset characteristics.

    Parameters
    ----------
    df:
        The loaded DataFrame.
    profile:
        Output of :func:`src.profiler.profile_dataset`.

    Returns
    -------
    List of recommendation dicts, each with keys:
    ``chart_type``, ``columns``, ``rationale``.
    """
    recs = []
    numeric_cols = profile["numeric_columns"]
    categorical_cols = profile["categorical_columns"]
    target_col = profile.get("likely_target_col")

    # Missing values bar chart
    any_missing = any(v > 0 for v in profile["missing_counts"].values())
    if any_missing:
        recs.append({
            "chart_type": "missing_values_bar",
            "columns": list(profile["missing_counts"].keys()),
            "rationale": "Visualise the extent of missingness across all columns.",
        })

    # Histograms for numeric columns
    for col in numeric_cols[:6]:
        recs.append({
            "chart_type": "numeric_histogram",
            "columns": [col],
            "rationale": f"Inspect the distribution of numeric column '{col}'.",
        })

    # Bar charts for categorical columns
    for col in categorical_cols[:4]:
        recs.append({
            "chart_type": "categorical_bar",
            "columns": [col],
            "rationale": f"Show the frequency of categories in '{col}'.",
        })

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        recs.append({
            "chart_type": "correlation_heatmap",
            "columns": numeric_cols,
            "rationale": "Reveal linear relationships between all numeric features.",
        })

        # Scatter plot for the strongest numeric pair
        try:
            corr = df[numeric_cols].corr().abs()
            # Zero out the diagonal
            import numpy as np
            np.fill_diagonal(corr.values, 0)
            max_idx = corr.stack().idxmax()
            recs.append({
                "chart_type": "scatter_plot",
                "columns": list(max_idx),
                "rationale": (
                    f"Scatter plot of the most correlated pair: "
                    f"'{max_idx[0]}' vs '{max_idx[1]}'."
                ),
            })
        except Exception:
            pass

    # Target-aware plots
    if target_col:
        if target_col in categorical_cols:
            recs.append({
                "chart_type": "target_distribution_bar",
                "columns": [target_col],
                "rationale": f"Show class balance for target column '{target_col}'.",
            })
            for col in numeric_cols[:3]:
                recs.append({
                    "chart_type": "box_plot_by_target",
                    "columns": [col, target_col],
                    "rationale": (
                        f"Box plot of '{col}' grouped by target '{target_col}'."
                    ),
                })
        elif target_col in numeric_cols:
            recs.append({
                "chart_type": "target_histogram",
                "columns": [target_col],
                "rationale": f"Distribution of numeric target '{target_col}'.",
            })

    return recs
