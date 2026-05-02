# Automated EDA Report: titanic_style

## Dataset Name
titanic_style

## Workflow Mode
llm__ollama_llama3.2_3b

## Selected Tools
- dataset_overview
- missing_value_analysis
- duplicate_analysis
- numeric_summary
- categorical_summary
- correlation_analysis
- outlier_detection
- visualization_recommendation
- chart_generation
- insight_generation

## LLM Planning Notes
Provider: ollama. Model: llama3.2:3b. Reasoning: Given the dataset has a mix of numerical and categorical columns, it's essential to perform an overview of the data, identify missing values, detect duplicates, summarize numeric and categorical columns, analyze correlations, detect outliers, and visualize key insights.

## Dataset Overview
Rows: 891, Columns: 11, Numeric: 7, Categorical: 4

## Data Quality Findings
Missing cells: 957 (9.76%).
Duplicate rows: 0 (0.00%).
Top missing columns:
| column      |   missing_count |   missing_percentage |
|-------------|-----------------|----------------------|
| Cabin       |             687 |            77.1044   |
| Age         |             186 |            20.8754   |
| Embarked    |              14 |             1.57127  |
| Fare        |              13 |             1.45903  |
| Pclass      |              10 |             1.12233  |
| Sex         |               9 |             1.0101   |
| Parch       |               9 |             1.0101   |
| PassengerId |               8 |             0.897868 |
| Name        |               7 |             0.785634 |
| SibSp       |               7 |             0.785634 |

## Statistical Findings
### Numeric Summary
| column   |      mean |       std |   min |     q1 |   median |     q3 |     max |   skewness |
|----------|-----------|-----------|-------|--------|----------|--------|---------|------------|
| Pclass   |  2.31328  |  0.833622 |  1    |  2     |   3      |  3     |   3     |  -0.639517 |
| Age      | 29.7371   | 14.5672   |  0.42 | 20.5   |  28      | 38     |  80     |   0.390346 |
| SibSp    |  0.511312 |  1.07103  |  0    |  0     |   0      |  1     |   8     |   3.66729  |
| Parch    |  0.38322  |  0.807507 |  0    |  0     |   0      |  0     |   6     |   2.7434   |
| Fare     | 32.454    | 49.9983   |  0    |  7.925 |  14.4583 | 31.275 | 512.329 |   4.74801  |
| Survived |  0.385747 |  0.487047 |  0    |  0     |   0      |  1     |   1     |   0.469434 |
### Categorical Summary
| column   |   unique_count | top_categories                      |
|----------|----------------|-------------------------------------|
| Sex      |              3 | male (571), female (311), <NA> (9)  |
| Embarked |              4 | S (634), C (166), Q (77)            |
| Cabin    |            148 | <NA> (687), G6 (4), C23 C25 C27 (4) |
### Correlation Highlights
Strongest pair: Pclass vs Fare (r=-0.549).
### Outlier Detection
| column   |   outlier_count |   outlier_percentage |   lower_bound |   upper_bound |
|----------|-----------------|----------------------|---------------|---------------|
| Parch    |             212 |             24.0363  |          0    |          0    |
| Fare     |             116 |             13.2118  |        -27.1  |         66.3  |
| SibSp    |              44 |              4.97738 |         -1.5  |          2.5  |
| Age      |              11 |              1.56028 |         -5.75 |         64.25 |
| Pclass   |               0 |              0       |          0.5  |          4.5  |
| Survived |               0 |              0       |         -1.5  |          2.5  |

## Key Insights
- Dataset contains 891 rows and 11 columns.
- Missing data affects 957 cells (9.76%); most impacted columns: Cabin (687), Age (186), Embarked (14).
- Pclass has mean 2.313, std 0.834, and spans [1.000, 3.000].
- Age has mean 29.737, std 14.567, and spans [0.420, 80.000].
- SibSp has mean 0.511, std 1.071, and spans [0.000, 8.000].
- Strongest absolute correlation is between Pclass and Fare (r=-0.549).
- Column Parch has the highest outlier load (212 rows, 24.04%).

## Visualization Recommendations
| chart_type          | x                | y                | title                         | reason                                       |   priority |
|---------------------|------------------|------------------|-------------------------------|----------------------------------------------|------------|
| correlation_heatmap | numeric_features | numeric_features | Correlation heatmap           | Multiple numeric columns are available.      |          1 |
| missing_bar         | column           | missing_count    | Missing values by column      | Columns with missing values were detected.   |          1 |
| scatter             | Pclass           | Fare             | Strongest numeric correlation | Useful for validating linear trend strength. |          1 |
| histogram           | Age              |                  | Distribution of Age           | Numeric distribution overview.               |          2 |
| histogram           | Fare             |                  | Distribution of Fare          | Numeric distribution overview.               |          2 |
| histogram           | Parch            |                  | Distribution of Parch         | Numeric distribution overview.               |          2 |
| histogram           | Pclass           |                  | Distribution of Pclass        | Numeric distribution overview.               |          2 |
| histogram           | SibSp            |                  | Distribution of SibSp         | Numeric distribution overview.               |          2 |
| target_box          | Survived         | Age              | Age by target (Survived)      | Highlights class-wise numeric spread.        |          3 |
| target_box          | Survived         | Pclass           | Pclass by target (Survived)   | Highlights class-wise numeric spread.        |          3 |

## Chart List
- charts\correlation_heatmap.png
- charts\missing_values_bar.png
- charts\scatter_Pclass_vs_Fare.png
- charts\hist_Age.png
- charts\hist_Fare.png
- charts\hist_Parch.png
- charts\hist_Pclass.png
- charts\hist_SibSp.png
- charts\target_box_Age_by_Survived.png
- charts\target_box_Pclass_by_Survived.png

## Limitations
- Results depend on heuristic target detection and may not match domain intent.
- Outlier logic is IQR-based and may over-flag skewed distributions.
- LLM mode can fall back to rules when planning output is invalid or unavailable.

## Reproducibility Notes
- Random seed: 42
- Charts generated with matplotlib only (no seaborn).
- Tool execution restricted to safe registry entries.
- Input data source: CSV files under data/sample/.
