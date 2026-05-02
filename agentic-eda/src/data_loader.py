"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd



def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file with default pandas parser and basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df



def list_sample_datasets(sample_dir: Path) -> list[Path]:
    """List all CSV datasets under the sample directory."""
    if not sample_dir.exists():
        return []
    return sorted(sample_dir.glob("*.csv"))
