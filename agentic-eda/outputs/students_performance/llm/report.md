# EDA Report: students_performance — llm mode

_Generated: 2026-04-30 02:26:06 UTC_

## Dataset Overview

| Property | Value |
|---|---|
| Dataset name | students_performance |
| Agent mode | llm |
| Rows | 1,000 |
| Columns | 10 |
| Numeric columns | 4 |
| Categorical columns | 6 |
| Datetime columns | 0 |
| Likely target column | grade |
| Memory usage (KB) | 360.16 |

**Numeric columns:** student_id, math_score, reading_score, writing_score

**Categorical columns:** gender, race_ethnicity, parental_education, lunch, test_preparation, grade

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
| student_id | 500.5000 | 500.5000 | 288.8194 | 1.0000 | 1000.0000 | 0.0 |
| math_score | 62.9540 | 63.0000 | 15.6428 | 1.0000 | 100.0000 | -0.206188 |
| reading_score | 63.7070 | 64.0000 | 16.6276 | 8.0000 | 100.0000 | -0.135288 |
| writing_score | 61.9800 | 62.0000 | 15.9396 | 0.0000 | 100.0000 | -0.177375 |

### Categorical Summary

**gender** — 2 unique values, mode: `female`

| Value | Count |
|---|---|
| female | 516 |
| male | 484 |

**race_ethnicity** — 5 unique values, mode: `group C`

| Value | Count |
|---|---|
| group C | 310 |
| group D | 262 |
| group B | 198 |
| group E | 139 |
| group A | 91 |

**parental_education** — 6 unique values, mode: `some college`

| Value | Count |
|---|---|
| some college | 231 |
| associate's degree | 212 |
| high school | 204 |
| bachelor's degree | 172 |
| some high school | 108 |

**lunch** — 2 unique values, mode: `standard`

| Value | Count |
|---|---|
| standard | 657 |
| free/reduced | 343 |

**test_preparation** — 2 unique values, mode: `none`

| Value | Count |
|---|---|
| none | 662 |
| completed | 338 |

**grade** — 5 unique values, mode: `C`

| Value | Count |
|---|---|
| C | 260 |
| D | 223 |
| B | 216 |
| F | 193 |
| A | 108 |

### Top Correlated Pairs

| Column 1 | Column 2 | Pearson r |
|---|---|---|
| math_score | writing_score | 0.7952 |
| math_score | reading_score | 0.7891 |
| reading_score | writing_score | 0.6170 |

### Outlier Detection (IQR method)

| Column | Outlier Count | Outlier % |
|---|---|---|
| student_id | 0 | 0.0% |
| math_score | 5 | 0.5% |
| reading_score | 6 | 0.6% |
| writing_score | 6 | 0.6% |

### Target-Aware Analysis

- Target column: **grade**
- Target type: categorical

**Class distribution:**

| Class | % |
|---|---|
| C | 26.0% |
| D | 22.3% |
| B | 21.6% |
| F | 19.3% |
| A | 10.8% |

## Key Insights

1. The dataset 'students_performance' contains 1,000 rows and 10 columns.
2. No missing values were detected; the dataset appears complete.
3. No duplicate rows were found.
4. Strongest correlation: 'math_score' ↔ 'writing_score' (r=0.795). Multicollinearity may affect linear models.
5. No numeric column has more than 5% IQR-flagged outliers.
6. Target 'grade' is relatively balanced across classes.
7. 13 visualizations were recommended to aid exploratory analysis.
8. No major data quality issues (missing values, duplicates, or heavy outliers) found.

## Visualization Recommendations

| # | Chart Type | Columns | Rationale |
|---|---|---|---|
| 1 | numeric_histogram | student_id | Inspect the distribution of numeric column 'student_id'. |
| 2 | numeric_histogram | math_score | Inspect the distribution of numeric column 'math_score'. |
| 3 | numeric_histogram | reading_score | Inspect the distribution of numeric column 'reading_score'. |
| 4 | numeric_histogram | writing_score | Inspect the distribution of numeric column 'writing_score'. |
| 5 | categorical_bar | gender | Show the frequency of categories in 'gender'. |
| 6 | categorical_bar | race_ethnicity | Show the frequency of categories in 'race_ethnicity'. |
| 7 | categorical_bar | parental_education | Show the frequency of categories in 'parental_education'. |
| 8 | categorical_bar | lunch | Show the frequency of categories in 'lunch'. |
| 9 | correlation_heatmap | student_id, math_score, reading_score, writing_score | Reveal linear relationships between all numeric features. |
| 10 | target_distribution_bar | grade | Show class balance for target column 'grade'. |
| 11 | box_plot_by_target | student_id, grade | Box plot of 'student_id' grouped by target 'grade'. |
| 12 | box_plot_by_target | math_score, grade | Box plot of 'math_score' grouped by target 'grade'. |
| 13 | box_plot_by_target | reading_score, grade | Box plot of 'reading_score' grouped by target 'grade'. |

## Generated Charts

- `hist_student_id.png`
- `hist_math_score.png`
- `hist_reading_score.png`
- `hist_writing_score.png`
- `catbar_gender.png`
- `catbar_race_ethnicity.png`
- `catbar_parental_education.png`
- `catbar_lunch.png`
- `correlation_heatmap.png`
- `target_dist_grade.png`
- `box_student_id_by_grade.png`
- `box_math_score_by_grade.png`
- `box_reading_score_by_grade.png`

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
