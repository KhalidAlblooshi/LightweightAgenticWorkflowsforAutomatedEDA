"""Dataset profiler — produces a rich metadata dict for downstream tools."""

from pathlib import Path

import pandas as pd

_TARGET_KEYWORDS = {
    "target", "label", "class", "price", "survived",
    "outcome", "grade", "score", "quality",
}


def profile_dataset(df: pd.DataFrame, dataset_name: str) -> dict:
    """Build a comprehensive profile of *df*.

    Parameters
    ----------
    df:
        The loaded DataFrame.
    dataset_name:
        Human-readable name used in reports.

    Returns
    -------
    dict with keys documented in the module spec.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pct = {
        col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns
    }
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    sample_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

    # Detect a likely target column
    likely_target_col = None
    for col in df.columns:
        if col.lower() in _TARGET_KEYWORDS:
            likely_target_col = col
            break

    return {
        "dataset_name": dataset_name,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "dtypes": dtypes,
        "sample_rows": sample_rows,
        "has_likely_target": likely_target_col is not None,
        "likely_target_col": likely_target_col,
    }
