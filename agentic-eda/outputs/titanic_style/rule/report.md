# EDA Report: titanic_style — rule mode

_Generated: 2026-04-30 02:26:09 UTC_

## Dataset Overview

| Property | Value |
|---|---|
| Dataset name | titanic_style |
| Agent mode | rule |
| Rows | 800 |
| Columns | 11 |
| Numeric columns | 7 |
| Categorical columns | 4 |
| Datetime columns | 0 |
| Likely target column | Survived |
| Memory usage (KB) | 212.94 |

**Numeric columns:** PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare

**Categorical columns:** Name, Sex, Cabin, Embarked

## Data Quality

### Missing Values

- Total missing cells: **779** (8.8%)
- Columns with >20% missing: Age, Cabin

| Column | Missing Count | Missing % |
|---|---|---|
| Age | 163 | 20.38% |
| Cabin | 615 | 76.88% |
| Embarked | 1 | 0.12% |

### Duplicates

- Duplicate rows: **0** (0.0%)

## Statistical Analysis

### Numeric Summary

| Column | Mean | Median | Std | Min | Max | Skewness |
|---|---|---|---|---|---|---|
| PassengerId | 400.5000 | 400.5000 | 231.0844 | 1.0000 | 800.0000 | 0.0 |
| Survived | 0.3850 | 0.0000 | 0.4869 | 0.0000 | 1.0000 | 0.47356 |
| Pclass | 2.3050 | 3.0000 | 0.8369 | 1.0000 | 3.0000 | -0.622347 |
| Age | 29.8711 | 28.0000 | 14.5443 | 0.6700 | 80.0000 | 0.396534 |
| SibSp | 0.5188 | 0.0000 | 1.0635 | 0.0000 | 8.0000 | 3.587 |
| Parch | 0.3738 | 0.0000 | 0.8015 | 0.0000 | 6.0000 | 2.777661 |
| Fare | 33.0385 | 14.5000 | 51.5249 | 0.0000 | 512.3292 | 4.706886 |

### Categorical Summary

**Name** — 800 unique values, mode: `Abbott, Mr. Rossmore Edward`

| Value | Count |
|---|---|
| Braund, Mr. Owen Harris | 1 |
| Cumings, Mrs. John Bradley (Florence Briggs Thayer) | 1 |
| Heikkinen, Miss. Laina | 1 |
| Futrelle, Mrs. Jacques Heath (Lily May Peel) | 1 |
| Allen, Mr. William Henry | 1 |

**Sex** — 2 unique values, mode: `male`

| Value | Count |
|---|---|
| male | 517 |
| female | 283 |

**Cabin** — 136 unique values, mode: `C23 C25 C27`

| Value | Count |
|---|---|
| G6 | 4 |
| C23 C25 C27 | 4 |
| F33 | 3 |
| E101 | 3 |
| F2 | 3 |

**Embarked** — 3 unique values, mode: `S`

| Value | Count |
|---|---|
| S | 577 |
| C | 149 |
| Q | 73 |

### Top Correlated Pairs

| Column 1 | Column 2 | Pearson r |
|---|---|---|
| Pclass | Fare | -0.5591 |
| SibSp | Parch | 0.4110 |
| Pclass | Age | -0.3722 |
| Survived | Pclass | -0.3254 |
| Age | SibSp | -0.3065 |

### Outlier Detection (IQR method)

| Column | Outlier Count | Outlier % |
|---|---|---|
| PassengerId | 0 | 0.0% |
| Survived | 0 | 0.0% |
| Pclass | 0 | 0.0% |
| Age | 12 | 1.88% |
| SibSp | 40 | 5.0% |
| Parch | 186 | 23.25% |
| Fare | 107 | 13.38% |

### Target-Aware Analysis

- Target column: **Survived**
- Target type: numeric

## Key Insights

1. The dataset 'titanic_style' contains 800 rows and 11 columns.
2. 779 missing cells detected (8.8% of all values).
3. Columns with >20% missing data: Age, Cabin. Consider imputation or removal before modelling.
4. No duplicate rows were found.
5. Highly skewed numeric columns (|skew|>1): 'SibSp' (skew=3.59), 'Parch' (skew=2.78), 'Fare' (skew=4.71). Log or power transforms may improve model performance.
6. Strongest correlation: 'Pclass' ↔ 'Fare' (r=-0.559). Multicollinearity may affect linear models.
7. 5 feature pairs have |r|>0.3, suggesting potential redundancy.
8. Columns with >5% outliers (IQR method): Parch, Fare. Investigate these rows and consider robust scalers.
9. High-cardinality categorical columns (>50 unique values): Name, Cabin. Encoding strategy should be chosen carefully.
10. Target 'Survived' is numeric; regression models are appropriate.
11. 13 visualizations were recommended to aid exploratory analysis.
12. Data quality issues found: missing values, outliers. Address these before downstream modelling.

## Visualization Recommendations

| # | Chart Type | Columns | Rationale |
|---|---|---|---|
| 1 | missing_values_bar | PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked | Visualise the extent of missingness across all columns. |
| 2 | numeric_histogram | PassengerId | Inspect the distribution of numeric column 'PassengerId'. |
| 3 | numeric_histogram | Survived | Inspect the distribution of numeric column 'Survived'. |
| 4 | numeric_histogram | Pclass | Inspect the distribution of numeric column 'Pclass'. |
| 5 | numeric_histogram | Age | Inspect the distribution of numeric column 'Age'. |
| 6 | numeric_histogram | SibSp | Inspect the distribution of numeric column 'SibSp'. |
| 7 | numeric_histogram | Parch | Inspect the distribution of numeric column 'Parch'. |
| 8 | categorical_bar | Name | Show the frequency of categories in 'Name'. |
| 9 | categorical_bar | Sex | Show the frequency of categories in 'Sex'. |
| 10 | categorical_bar | Cabin | Show the frequency of categories in 'Cabin'. |
| 11 | categorical_bar | Embarked | Show the frequency of categories in 'Embarked'. |
| 12 | correlation_heatmap | PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare | Reveal linear relationships between all numeric features. |
| 13 | target_histogram | Survived | Distribution of numeric target 'Survived'. |

## Generated Charts

- `missing_values_bar.png`
- `hist_PassengerId.png`
- `hist_Survived.png`
- `hist_Pclass.png`
- `hist_Age.png`
- `hist_SibSp.png`
- `hist_Parch.png`
- `catbar_Name.png`
- `catbar_Sex.png`
- `catbar_Cabin.png`
- `catbar_Embarked.png`
- `correlation_heatmap.png`
- `target_dist_Survived.png`

## Limitations

- This report was generated automatically and may not capture all nuances of the data.
- Statistical tests (normality, significance) are not performed.
- Insights are heuristic and should be validated by a domain expert.
- LLM-based tool selection may vary across runs.
- Visualizations are limited to the most informative subset of columns.

## Reproducibility Notes

- Random seed: 42
- All synthetic data generated with `numpy.random.seed(42)`.
- Agent mode: **rule**
- To reproduce: `python main.py --dataset <path> --mode rule`
