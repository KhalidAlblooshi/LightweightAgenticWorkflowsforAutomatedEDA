# EDA Report: wine_quality — llm mode

_Generated: 2026-04-30 02:26:14 UTC_

## Dataset Overview

| Property | Value |
|---|---|
| Dataset name | wine_quality |
| Agent mode | llm |
| Rows | 1,200 |
| Columns | 13 |
| Numeric columns | 12 |
| Categorical columns | 1 |
| Datetime columns | 0 |
| Likely target column | quality |
| Memory usage (KB) | 174.85 |

**Numeric columns:** fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality

**Categorical columns:** wine_type

## Data Quality

### Missing Values

- Total missing cells: **0** (0.0%)
- No missing values found.

### Duplicates

- Duplicate rows: **0** (0.0%)

## Statistical Analysis

### Numeric Summary

| Column | Mean | Median | Std | Min | Max | Skewness |
|---|---|---|---|---|---|---|
| fixed_acidity | 7.5297 | 7.2950 | 1.4389 | 4.3300 | 13.1000 | 0.783406 |
| volatile_acidity | 0.3950 | 0.3600 | 0.1890 | -0.0220 | 1.2770 | 0.721249 |
| citric_acid | 0.3095 | 0.3145 | 0.1528 | 0.0000 | 0.8150 | 0.055875 |
| residual_sugar | 4.7740 | 2.7700 | 5.3785 | 0.9000 | 45.9000 | 2.656113 |
| chlorides | 0.0678 | 0.0585 | 0.0426 | 0.0100 | 0.2246 | 0.87425 |
| free_sulfur_dioxide | 26.1871 | 23.4000 | 16.7848 | 1.0000 | 72.0000 | 0.57116 |
| total_sulfur_dioxide | 103.0907 | 89.2000 | 69.2873 | 6.0000 | 290.0000 | 0.747748 |
| density | 0.9954 | 0.9956 | 0.0028 | 0.9900 | 1.0037 | -0.125195 |
| pH | 3.2455 | 3.2440 | 0.1609 | 2.7740 | 3.8620 | 0.077088 |
| sulphates | 0.5687 | 0.5440 | 0.1627 | 0.3300 | 1.1890 | 0.726185 |
| alcohol | 10.4354 | 10.4050 | 1.2099 | 8.0000 | 14.5700 | 0.142553 |
| quality | 5.4892 | 5.0000 | 0.6894 | 4.0000 | 8.0000 | 0.031289 |

### Categorical Summary

**wine_type** — 2 unique values, mode: `white`

| Value | Count |
|---|---|
| white | 656 |
| red | 544 |

### Top Correlated Pairs

| Column 1 | Column 2 | Pearson r |
|---|---|---|
| free_sulfur_dioxide | total_sulfur_dioxide | 0.9259 |
| alcohol | quality | 0.5335 |
| volatile_acidity | free_sulfur_dioxide | -0.4070 |
| volatile_acidity | total_sulfur_dioxide | -0.3769 |
| chlorides | free_sulfur_dioxide | -0.3538 |

### Outlier Detection (IQR method)

| Column | Outlier Count | Outlier % |
|---|---|---|
| fixed_acidity | 24 | 2.0% |
| volatile_acidity | 11 | 0.92% |
| citric_acid | 7 | 0.58% |
| residual_sugar | 76 | 6.33% |
| chlorides | 26 | 2.17% |
| free_sulfur_dioxide | 0 | 0.0% |
| total_sulfur_dioxide | 20 | 1.67% |
| density | 1 | 0.08% |
| pH | 11 | 0.92% |
| sulphates | 10 | 0.83% |
| alcohol | 4 | 0.33% |
| quality | 2 | 0.17% |

### Target-Aware Analysis

- Target column: **quality**
- Target type: numeric

## Key Insights

1. The dataset 'wine_quality' contains 1,200 rows and 13 columns.
2. No missing values were detected; the dataset appears complete.
3. No duplicate rows were found.
4. Highly skewed numeric columns (|skew|>1): 'residual_sugar' (skew=2.66). Log or power transforms may improve model performance.
5. Strongest correlation: 'free_sulfur_dioxide' ↔ 'total_sulfur_dioxide' (r=0.926). Multicollinearity may affect linear models.
6. 5 feature pairs have |r|>0.3, suggesting potential redundancy.
7. Columns with >5% outliers (IQR method): residual_sugar. Investigate these rows and consider robust scalers.
8. Target 'quality' is numeric; regression models are appropriate.
9. 9 visualizations were recommended to aid exploratory analysis.
10. Data quality issues found: outliers. Address these before downstream modelling.

## Visualization Recommendations

| # | Chart Type | Columns | Rationale |
|---|---|---|---|
| 1 | numeric_histogram | fixed_acidity | Inspect the distribution of numeric column 'fixed_acidity'. |
| 2 | numeric_histogram | volatile_acidity | Inspect the distribution of numeric column 'volatile_acidity'. |
| 3 | numeric_histogram | citric_acid | Inspect the distribution of numeric column 'citric_acid'. |
| 4 | numeric_histogram | residual_sugar | Inspect the distribution of numeric column 'residual_sugar'. |
| 5 | numeric_histogram | chlorides | Inspect the distribution of numeric column 'chlorides'. |
| 6 | numeric_histogram | free_sulfur_dioxide | Inspect the distribution of numeric column 'free_sulfur_dioxide'. |
| 7 | categorical_bar | wine_type | Show the frequency of categories in 'wine_type'. |
| 8 | correlation_heatmap | fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality | Reveal linear relationships between all numeric features. |
| 9 | target_histogram | quality | Distribution of numeric target 'quality'. |

## Generated Charts

- `hist_fixed_acidity.png`
- `hist_volatile_acidity.png`
- `hist_citric_acid.png`
- `hist_residual_sugar.png`
- `hist_chlorides.png`
- `hist_free_sulfur_dioxide.png`
- `catbar_wine_type.png`
- `correlation_heatmap.png`
- `target_dist_quality.png`

## Limitations

- This report was generated automatically and may not capture all nuances of the data.
- Statistical tests (normality, significance) are not performed.
- Insights are heuristic and should be validated by a domain expert.
- LLM-based tool selection may vary across runs.
- Visualizations are limited to the most informative subset of columns.

## Reproducibility Notes

- Random seed: 42
- All synthetic data generated with `numpy.random.seed(42)`.
- Agent mode: **llm**
- To reproduce: `python main.py --dataset <path> --mode llm`
