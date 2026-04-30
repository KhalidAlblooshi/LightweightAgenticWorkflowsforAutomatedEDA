"""Data loading helpers."""

from pathlib import Path

import pandas as pd

from config import config


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and validate it is non-empty.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    ValueError
        When the file is empty or cannot be parsed as a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV at {path}: {exc}") from exc
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty (0 rows or 0 columns).")
    return df


def list_sample_datasets() -> list:
    """Return a sorted list of all CSV paths inside DATA_DIR."""
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))
