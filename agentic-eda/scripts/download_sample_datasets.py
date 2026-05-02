"""Download sample datasets, with deterministic synthetic fallbacks when offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RNG = np.random.default_rng(42)



def _safe_get_csv(url: str, sep: str = ",", timeout: int = 20) -> pd.DataFrame | None:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        from io import StringIO

        return pd.read_csv(StringIO(response.text), sep=sep)
    except Exception:
        return None



def _inject_missing_values(df: pd.DataFrame, frac: float = 0.03) -> pd.DataFrame:
    out = df.copy()
    n_rows, n_cols = out.shape
    n_cells = int(n_rows * n_cols * frac)
    if n_cells <= 0:
        return out

    for _ in range(n_cells):
        r = RNG.integers(0, n_rows)
        c = RNG.integers(0, n_cols)
        out.iat[r, c] = np.nan
    return out



def _make_titanic_synthetic(n_rows: int = 900) -> pd.DataFrame:
    passenger_id = np.arange(1, n_rows + 1)
    pclass = RNG.choice([1, 2, 3], size=n_rows, p=[0.24, 0.21, 0.55])
    sex = RNG.choice(["male", "female"], size=n_rows, p=[0.64, 0.36])
    age = np.clip(RNG.normal(30, 14, n_rows), 0.4, 80)
    sibsp = RNG.poisson(0.5, n_rows)
    parch = RNG.poisson(0.4, n_rows)
    embarked = RNG.choice(["S", "C", "Q"], size=n_rows, p=[0.72, 0.18, 0.10])
    fare_base = np.where(pclass == 1, 75, np.where(pclass == 2, 30, 12))
    fare = np.clip(fare_base + RNG.normal(0, 10, n_rows), 4, None)

    survival_logit = (
        -1.2
        + 0.9 * (sex == "female").astype(float)
        + 0.45 * (pclass == 1).astype(float)
        - 0.015 * age
        + 0.005 * fare
    )
    survival_prob = 1 / (1 + np.exp(-survival_logit))
    survived = (RNG.uniform(0, 1, n_rows) < survival_prob).astype(int)

    df = pd.DataFrame(
        {
            "PassengerId": passenger_id,
            "Pclass": pclass,
            "Name": [f"Passenger_{i}" for i in passenger_id],
            "Sex": sex,
            "Age": age.round(1),
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare.round(2),
            "Embarked": embarked,
            "Survived": survived,
            "Cabin": RNG.choice(["C85", "E31", "B57", "D26", ""], size=n_rows, p=[0.10, 0.10, 0.08, 0.07, 0.65]),
        }
    )
    return _inject_missing_values(df, frac=0.02)



def _make_diamonds_synthetic(n_rows: int = 1200) -> pd.DataFrame:
    carat = np.clip(RNG.normal(1.0, 0.7, n_rows), 0.2, 6.0).round(3)
    cut = RNG.choice(["Fair", "Good", "Very Good", "Premium", "Ideal"], size=n_rows, p=[0.05, 0.25, 0.30, 0.20, 0.20])
    color = RNG.choice(["D", "E", "F", "G", "H", "I", "J"], size=n_rows)
    clarity = RNG.choice(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], size=n_rows)
    depth = np.clip(RNG.normal(61, 1.5, n_rows), 54, 66).round(2)
    table = np.clip(RNG.normal(57, 2.0, n_rows), 50, 70).round(2)

    # simple proxy for price: roughly proportional to carat, with noise and modifiers
    price = np.clip((carat ** 1.9) * 4000 + RNG.normal(0, 1000, n_rows), 300, None).round(0).astype(int)

    # approximate dimensions (not physically accurate, but consistent)
    x = np.clip(5.5 * (carat ** (1 / 3)) + RNG.normal(0, 0.3, n_rows), 2.0, 10.0).round(2)
    y = np.clip(x + RNG.normal(0, 0.15, n_rows), 2.0, 10.0).round(2)
    z = np.clip((depth / 100) * (x + y) / 2 + RNG.normal(0, 0.1, n_rows), 1.0, 6.0).round(2)

    df = pd.DataFrame(
        {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "price": price,
            "x": x,
            "y": y,
            "z": z,
        }
    )
    return _inject_missing_values(df, frac=0.01)



def _make_adult_synthetic(n_rows: int = 2000) -> pd.DataFrame:
    age = RNG.integers(17, 90, size=n_rows)
    workclass = RNG.choice(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], size=n_rows, p=[0.7,0.07,0.03,0.03,0.05,0.04,0.005,0.005])
    fnlwgt = RNG.integers(10000, 800000, size=n_rows)
    education = RNG.choice(["Bachelors","Some-college","11th","HS-grad","Masters","9th","Assoc-acdm","Assoc-voc","7th-8th","Doctorate","Prof-school","5th-6th","10th","1st-4th","Preschool"], size=n_rows)
    education_num = np.array([{
        "Preschool":1,"1st-4th":2,"5th-6th":3,"7th-8th":4,"9th":5,"10th":6,"11th":7,"HS-grad":8,"Some-college":9,
        "Assoc-voc":10,"Assoc-acdm":11,"Bachelors":12,"Masters":13,"Prof-school":14,"Doctorate":15
    }[e] for e in education])
    marital_status = RNG.choice(["Married-civ-spouse","Divorced","Never-married","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"], size=n_rows)
    occupation = RNG.choice(["Tech-support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"], size=n_rows)
    relationship = RNG.choice(["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"], size=n_rows)
    race = RNG.choice(["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"], size=n_rows)
    sex = RNG.choice(["Male","Female"], size=n_rows, p=[0.65,0.35])
    capital_gain = RNG.choice([0,0,0,0,0, 1000, 5000, 99999], size=n_rows, p=[0.85,0.03,0.02,0.02,0.03,0.03,0.01,0.01])
    capital_loss = RNG.choice([0,0,0,0,0,100,400,2000], size=n_rows, p=[0.9,0.02,0.02,0.02,0.02,0.01,0.005,0.005])
    hours_per_week = RNG.integers(1, 99, size=n_rows)
    native_country = RNG.choice(["United-States","Mexico","Philippines","Germany","Canada","Puerto-Rico","Honduras","India","China","England","Other"], size=n_rows, p=[0.75,0.03,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.09])

    # simple income label
    income_score = (education_num > 10).astype(int) + (hours_per_week > 40).astype(int) + (capital_gain > 0).astype(int) + (age > 30).astype(int)
    income = np.where(income_score >= 2, ">50K", "<=50K")

    df = pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": native_country,
            "income": income,
        }
    )
    return _inject_missing_values(df, frac=0.01)



def _prepare_titanic() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    downloaded = _safe_get_csv(url)
    if downloaded is not None and not downloaded.empty:
        df = downloaded.copy()
        keep_cols = [
            col
            for col in ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived", "Cabin"]
            if col in df.columns
        ]
        df = df[keep_cols]
        return _inject_missing_values(df, frac=0.01)
    return _make_titanic_synthetic()



def _prepare_diamonds() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
    downloaded = _safe_get_csv(url)
    if downloaded is not None and not downloaded.empty:
        df = downloaded.copy()
        # ensure expected columns exist and correct types
        expected = {"carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"}
        if expected.issubset(set(df.columns)):
            # coerce numeric types
            for col in ["carat", "depth", "table", "price", "x", "y", "z"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return _inject_missing_values(df, frac=0.005)
    return _make_diamonds_synthetic()



def _prepare_adult() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    downloaded = _safe_get_csv(url, sep=",", timeout=30)
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    if downloaded is not None and not downloaded.empty:
        # adult.data has no header and may contain trailing spaces; reload accordingly
        try:
            from io import StringIO

            raw = downloaded.to_csv(index=False, header=False)
            df = pd.read_csv(StringIO(raw), names=col_names, sep=",", skipinitialspace=True, na_values=["?", " ?"])
        except Exception:
            df = downloaded.copy()
        # clean income field and strip whitespace
        if "income" in df.columns:
            df["income"] = df["income"].astype(str).str.strip().replace({"<=50K.": "<=50K", ">50K.": ">50K"})
        return _inject_missing_values(df, frac=0.01)

    return _make_adult_synthetic()



def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "titanic_style.csv": _prepare_titanic(),
        "diamonds.csv": _prepare_diamonds(),
        "adult_income.csv": _prepare_adult(),
    }

    print("Creating sample datasets...")
    for filename, frame in datasets.items():
        output_path = SAMPLE_DIR / filename
        frame.to_csv(output_path, index=False)
        print(f"- {output_path} | rows={len(frame)} cols={len(frame.columns)}")

    print("Done.")



if __name__ == "__main__":
    main()
