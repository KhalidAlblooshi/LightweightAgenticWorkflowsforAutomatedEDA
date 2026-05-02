# Automated EDA Report: adult_income

## Dataset Name
adult_income

## Workflow Mode
llm__ollama_qwen2.5_3b

## Selected Tools
- dataset_overview
- missing_value_analysis
- duplicate_analysis
- categorical_summary
- correlation_analysis
- visualization_recommendation
- chart_generation
- insight_generation

## LLM Planning Notes
Provider: ollama. Model: qwen2.5:3b. Reasoning: The dataset contains a mix of numerical and categorical data. The 'income' column is likely the target variable as it represents the income level which can be binary (<=50K or >50K). We will start by understanding the overall structure with `dataset_overview`, then check for missing values, duplicates, and distributions.

## Dataset Overview
Rows: 32560, Columns: 15, Numeric: 6, Categorical: 9

## Data Quality Findings
Missing cells: 9061 (1.86%).
Duplicate rows: 19 (0.06%).
Top missing columns:
| column         |   missing_count |   missing_percentage |
|----------------|-----------------|----------------------|
| occupation     |            2143 |             6.5817   |
| workclass      |            2127 |             6.53256  |
| native_country |             903 |             2.77334  |
| education_num  |             364 |             1.11794  |
| fnlwgt         |             351 |             1.07801  |
| age            |             338 |             1.03808  |
| sex            |             336 |             1.03194  |
| race           |             331 |             1.01658  |
| capital_gain   |             330 |             1.01351  |
| relationship   |             318 |             0.976658 |

## Statistical Findings
### Categorical Summary
| column         |   unique_count | top_categories                                                     |
|----------------|----------------|--------------------------------------------------------------------|
| workclass      |              9 | Private (22479), Self-emp-not-inc (2518), <NA> (2127)              |
| education      |             17 | HS-grad (10402), Some-college (7232), Bachelors (5304)             |
| marital_status |              8 | Married-civ-spouse (14820), Never-married (10579), Divorced (4410) |
| occupation     |             15 | Prof-specialty (4100), Craft-repair (4061), Exec-managerial (4023) |
| relationship   |              7 | Husband (13058), Not-in-family (8225), Own-child (5024)            |
| race           |              6 | White (27522), Black (3096), Asian-Pac-Islander (1033)             |
### Correlation Highlights
Strongest pair: education_num vs hours_per_week (r=0.148).

## Key Insights
- Dataset contains 32560 rows and 15 columns.
- Missing data affects 9061 cells (1.86%); most impacted columns: occupation (2143), workclass (2127), native_country (903).
- Detected 19 duplicate rows (0.06% of dataset), which may bias model evaluation if not handled.
- Strongest absolute correlation is between education_num and hours_per_week (r=0.148).

## Visualization Recommendations
| chart_type          | x                | y                | title                                 | reason                                       |   priority |
|---------------------|------------------|------------------|---------------------------------------|----------------------------------------------|------------|
| correlation_heatmap | numeric_features | numeric_features | Correlation heatmap                   | Multiple numeric columns are available.      |          1 |
| missing_bar         | column           | missing_count    | Missing values by column              | Columns with missing values were detected.   |          1 |
| scatter             | education_num    | hours_per_week   | Strongest numeric correlation         | Useful for validating linear trend strength. |          1 |
| bar                 | income           | count            | Category frequencies for income       | Categorical frequency comparison.            |          2 |
| bar                 | race             | count            | Category frequencies for race         | Categorical frequency comparison.            |          2 |
| bar                 | relationship     | count            | Category frequencies for relationship | Categorical frequency comparison.            |          2 |
| bar                 | sex              | count            | Category frequencies for sex          | Categorical frequency comparison.            |          2 |
| histogram           | age              |                  | Distribution of age                   | Numeric distribution overview.               |          2 |
| histogram           | capital_gain     |                  | Distribution of capital_gain          | Numeric distribution overview.               |          2 |
| histogram           | capital_loss     |                  | Distribution of capital_loss          | Numeric distribution overview.               |          2 |
| histogram           | education_num    |                  | Distribution of education_num         | Numeric distribution overview.               |          2 |
| histogram           | hours_per_week   |                  | Distribution of hours_per_week        | Numeric distribution overview.               |          2 |
| target_box          | income           | education_num    | education_num by target (income)      | Highlights class-wise numeric spread.        |          3 |
| target_box          | income           | age              | age by target (income)                | Highlights class-wise numeric spread.        |          3 |

## Chart List
- charts\correlation_heatmap.png
- charts\missing_values_bar.png
- charts\scatter_education_num_vs_hours_per_week.png
- charts\catbar_income.png
- charts\catbar_race.png
- charts\catbar_relationship.png
- charts\catbar_sex.png
- charts\hist_age.png
- charts\hist_capital_gain.png
- charts\hist_capital_loss.png
- charts\hist_education_num.png
- charts\hist_hours_per_week.png
- charts\target_box_education_num_by_income.png
- charts\target_box_age_by_income.png

## Limitations
- Results depend on heuristic target detection and may not match domain intent.
- Outlier logic is IQR-based and may over-flag skewed distributions.
- LLM mode can fall back to rules when planning output is invalid or unavailable.

## Reproducibility Notes
- Random seed: 42
- Charts generated with matplotlib only (no seaborn).
- Tool execution restricted to safe registry entries.
- Input data source: CSV files under data/sample/.
