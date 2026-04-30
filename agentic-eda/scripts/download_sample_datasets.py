"""Download or generate sample datasets for agentic-eda."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve paths relative to this script's location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data" / "sample"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def _titanic_style(n: int = 800) -> pd.DataFrame:
    """Generate a Titanic-style dataset with realistic correlations."""
    pclass = RNG.choice([1, 2, 3], size=n, p=[0.25, 0.30, 0.45])
    sex = RNG.choice(["male", "female"], size=n, p=[0.65, 0.35])

    # Age: correlated with class (higher class → slightly older passengers)
    base_age = np.where(pclass == 1, 38, np.where(pclass == 2, 32, 26))
    age = base_age + RNG.normal(0, 12, n)
    age = np.clip(age, 1, 80).astype(float)
    # ~15% missing
    age[RNG.random(n) < 0.15] = np.nan

    # Fare: strongly correlated with class
    base_fare = np.where(pclass == 1, 85, np.where(pclass == 2, 22, 10))
    fare = base_fare * RNG.lognormal(0, 0.5, n)
    fare = np.clip(fare, 3.5, 600).round(2)

    sibsp = RNG.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.28, 0.10, 0.05, 0.02])
    parch = RNG.choice([0, 1, 2, 3], size=n, p=[0.65, 0.20, 0.10, 0.05])

    # Survival: correlated with sex, pclass, age
    p_survive = (
        0.35
        + 0.30 * (sex == "female")
        - 0.15 * (pclass == 3)
        + 0.10 * (pclass == 1)
    )
    p_survive = np.clip(p_survive, 0.05, 0.95)
    survived = (RNG.random(n) < p_survive).astype(int)

    # Embarked: mostly S
    embarked = RNG.choice(["S", "C", "Q"], size=n, p=[0.72, 0.20, 0.08])
    embarked_series = pd.array(embarked, dtype=object)
    embarked_series[RNG.random(n) < 0.02] = None

    # Cabin: mostly missing
    cabin_letters = ["A", "B", "C", "D", "E", "F"]
    cabin = np.array([
        f"{RNG.choice(cabin_letters)}{RNG.integers(1, 150)}" for _ in range(n)
    ], dtype=object)
    cabin[RNG.random(n) < 0.70] = None

    names = [f"Passenger_{i:04d}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked_series,
    })
    return df


def _students_performance(n: int = 1000) -> pd.DataFrame:
    """Generate a student performance dataset."""
    gender = RNG.choice(["male", "female"], size=n, p=[0.48, 0.52])
    race = RNG.choice(["group A", "group B", "group C", "group D", "group E"],
                      size=n, p=[0.10, 0.19, 0.32, 0.26, 0.13])
    parent_ed = RNG.choice(
        ["some high school", "high school", "some college",
         "associate's degree", "bachelor's degree", "master's degree"],
        size=n,
        p=[0.10, 0.20, 0.23, 0.22, 0.18, 0.07],
    )
    lunch = RNG.choice(["standard", "free/reduced"], size=n, p=[0.65, 0.35])
    test_prep = RNG.choice(["none", "completed"], size=n, p=[0.64, 0.36])

    # Scores correlated with lunch, test prep, parent education
    ed_boost = np.where(
        parent_ed == "master's degree", 8,
        np.where(parent_ed == "bachelor's degree", 5,
                 np.where(parent_ed == "associate's degree", 2, 0))
    )
    lunch_boost = np.where(lunch == "standard", 5, 0)
    prep_boost = np.where(test_prep == "completed", 7, 0)

    base_math = 55 + ed_boost + lunch_boost + prep_boost
    math_score = np.clip(base_math + RNG.normal(0, 15, n), 0, 100).round().astype(int)

    # Reading and writing correlated with math
    reading_score = np.clip(
        math_score * 0.85 + RNG.normal(10, 10, n), 0, 100
    ).round().astype(int)
    writing_score = np.clip(
        math_score * 0.80 + RNG.normal(12, 10, n), 0, 100
    ).round().astype(int)

    avg_score = (math_score + reading_score + writing_score) / 3
    grade = pd.cut(
        avg_score,
        bins=[0, 50, 60, 70, 80, 100],
        labels=["F", "D", "C", "B", "A"],
        right=True,
    )

    df = pd.DataFrame({
        "student_id": range(1, n + 1),
        "gender": gender,
        "race_ethnicity": race,
        "parental_education": parent_ed,
        "lunch": lunch,
        "test_preparation": test_prep,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
        "grade": grade.astype(str),
    })
    return df


def _wine_quality(n: int = 1200) -> pd.DataFrame:
    """Generate a wine quality dataset."""
    wine_type = RNG.choice(["red", "white"], size=n, p=[0.45, 0.55])

    # Physical properties with type-based differences
    is_red = wine_type == "red"

    fixed_acidity = np.where(is_red,
                             RNG.normal(8.5, 1.5, n),
                             RNG.normal(6.8, 0.8, n))

    volatile_acidity = np.where(is_red,
                                RNG.normal(0.53, 0.18, n),
                                RNG.normal(0.28, 0.10, n))

    citric_acid = np.clip(
        np.where(is_red, RNG.normal(0.27, 0.19, n), RNG.normal(0.33, 0.12, n)),
        0, 1
    )

    residual_sugar = np.clip(
        np.where(is_red, RNG.exponential(2.5, n), RNG.exponential(6.5, n)),
        0.9, 65
    )

    chlorides = np.clip(
        np.where(is_red, RNG.normal(0.087, 0.047, n), RNG.normal(0.045, 0.022, n)),
        0.01, 0.6
    )

    free_so2 = np.clip(
        np.where(is_red, RNG.normal(15, 10, n), RNG.normal(35, 17, n)),
        1, 72
    )

    total_so2 = np.clip(free_so2 * RNG.uniform(2.5, 5.5, n), 6, 290)

    density = np.clip(
        np.where(is_red, RNG.normal(0.9967, 0.002, n), RNG.normal(0.9940, 0.003, n)),
        0.99, 1.004
    )

    ph = np.clip(
        np.where(is_red, RNG.normal(3.31, 0.15, n), RNG.normal(3.19, 0.15, n)),
        2.74, 4.01
    )

    sulphates = np.clip(
        np.where(is_red, RNG.normal(0.66, 0.17, n), RNG.normal(0.49, 0.11, n)),
        0.33, 2.0
    )

    alcohol = np.clip(
        RNG.normal(10.4, 1.2, n), 8.0, 14.9
    )

    # Quality: correlated with alcohol, sulphates, low volatile acidity
    quality_score = (
        5.5
        + 0.3 * (alcohol - 10.4)
        + 0.5 * (sulphates - 0.6)
        - 1.0 * (volatile_acidity - 0.4)
        + RNG.normal(0, 0.5, n)
    )
    quality = np.clip(quality_score.round(), 3, 9).astype(int)

    df = pd.DataFrame({
        "fixed_acidity": fixed_acidity.round(2),
        "volatile_acidity": volatile_acidity.round(3),
        "citric_acid": citric_acid.round(3),
        "residual_sugar": residual_sugar.round(2),
        "chlorides": chlorides.round(4),
        "free_sulfur_dioxide": free_so2.round(1),
        "total_sulfur_dioxide": total_so2.round(1),
        "density": density.round(6),
        "pH": ph.round(3),
        "sulphates": sulphates.round(3),
        "alcohol": alcohol.round(2),
        "quality": quality,
        "wine_type": wine_type,
    })
    return df


# ---------------------------------------------------------------------------
# Download or generate each dataset
# ---------------------------------------------------------------------------

def _try_download_titanic() -> pd.DataFrame | None:
    """Try to download the Titanic dataset from a public source."""
    try:
        import requests
        url = (
            "https://raw.githubusercontent.com/datasciencedojo/datasets/"
            "master/titanic.csv"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        from io import StringIO
        return pd.read_csv(StringIO(resp.text))
    except Exception:
        return None


def save_dataset(df: pd.DataFrame, name: str) -> Path:
    path = _DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"  Saved {name} ({len(df):,} rows × {len(df.columns)} cols) → {path}")
    return path


def main():
    print("Downloading / generating sample datasets...\n")

    # --- Titanic-style ---
    print("[1/3] Titanic-style dataset")
    titanic_df = _try_download_titanic()
    if titanic_df is not None and len(titanic_df) > 0:
        # Rename columns to match spec
        col_map = {
            "PassengerId": "PassengerId", "Survived": "Survived",
            "Pclass": "Pclass", "Name": "Name", "Sex": "Sex",
            "Age": "Age", "SibSp": "SibSp", "Parch": "Parch",
            "Fare": "Fare", "Cabin": "Cabin", "Embarked": "Embarked",
        }
        titanic_df = titanic_df[[c for c in col_map if c in titanic_df.columns]]
        # Use only first 800 rows or pad with synthetic if fewer
        if len(titanic_df) < 800:
            synth = _titanic_style(800 - len(titanic_df))
            titanic_df = pd.concat([titanic_df, synth], ignore_index=True)
        else:
            titanic_df = titanic_df.head(800)
        print("  (Downloaded real Titanic data)")
    else:
        print("  (Download failed — generating synthetic data)")
        titanic_df = _titanic_style(800)
    save_dataset(titanic_df, "titanic_style.csv")

    # --- Students performance ---
    print("\n[2/3] Students performance dataset")
    students_df = _students_performance(1000)
    save_dataset(students_df, "students_performance.csv")

    # --- Wine quality ---
    print("\n[3/3] Wine quality dataset")
    wine_df = _wine_quality(1200)
    save_dataset(wine_df, "wine_quality.csv")

    print(f"\nAll datasets saved to: {_DATA_DIR}")


if __name__ == "__main__":
    main()
