# Automated EDA Report: adult_income

## Dataset Name
adult_income

## Workflow Mode
llm__ollama_tinyllama_latest

## Selected Tools
- dataset_overview
- missing_value_analysis
- duplicate_analysis
- numeric_summary
- correlation_analysis
- outlier_detection
- categorical_summary
- target_aware_analysis
- visualization_recommendation
- chart_generation
- insight_generation

## LLM Planning Notes
LLM planning failed. Fallback to rule-based workflow. Reason: LLM response did not contain JSON object.

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
### Numeric Summary
| column         |        mean |          std |   min |     q1 |   median |     q3 |            max |   skewness |
|----------------|-------------|--------------|-------|--------|----------|--------|----------------|------------|
| age            |     38.58   |     13.639   |    17 |     28 |       37 |     48 |    90          |   0.559531 |
| fnlwgt         | 189776      | 105561       | 12285 | 117814 |   178353 | 237051 |     1.4847e+06 |   1.44958  |
| education_num  |     10.0781 |      2.57214 |     1 |      9 |       10 |     12 |    16          |  -0.312843 |
| capital_gain   |   1072.87   |   7356.67    |     0 |      0 |        0 |      0 | 99999          |  11.9913   |
| capital_loss   |     87.1841 |    402.83    |     0 |      0 |        0 |      0 |  4356          |   4.60026  |
| hours_per_week |     40.4426 |     12.3583  |     1 |     40 |       40 |     45 |    99          |   0.227062 |
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
### Outlier Detection
| column         |   outlier_count |   outlier_percentage |   lower_bound |   upper_bound |
|----------------|-----------------|----------------------|---------------|---------------|
| hours_per_week |            8922 |            27.6591   |          32.5 |          52.5 |
| capital_gain   |            2683 |             8.32454  |           0   |           0   |
| capital_loss   |            1502 |             4.65635  |           0   |           0   |
| education_num  |            1185 |             3.68058  |           4.5 |          16.5 |
| fnlwgt         |             980 |             3.04263  |      -61041.5 |      415906   |
| age            |             143 |             0.443796 |          -2   |          78   |
### Target-Aware Analysis
Target column: income
- Target class distribution: <=50K=24499, >50K=7767.
- ANOVA for 'age' across target 'income' groups: p-value=0.
- ANOVA for 'fnlwgt' across target 'income' groups: p-value=0.1272.
- ANOVA for 'education_num' across target 'income' groups: p-value=0.
- ANOVA for 'capital_gain' across target 'income' groups: p-value=0.
- ANOVA for 'capital_loss' across target 'income' groups: p-value=3.442e-159.
- ANOVA for 'hours_per_week' across target 'income' groups: p-value=0.

## Key Insights
- Dataset contains 32560 rows and 15 columns.
- Missing data affects 9061 cells (1.86%); most impacted columns: occupation (2143), workclass (2127), native_country (903).
- Detected 19 duplicate rows (0.06% of dataset), which may bias model evaluation if not handled.
- age has mean 38.580, std 13.639, and spans [17.000, 90.000].
- fnlwgt has mean 189776.362, std 105561.414, and spans [12285.000, 1484705.000].
- education_num has mean 10.078, std 2.572, and spans [1.000, 16.000].
- Strongest absolute correlation is between education_num and hours_per_week (r=0.148).
- Column hours_per_week has the highest outlier load (8922 rows, 27.66%).
- Target class distribution: <=50K=24499, >50K=7767.
- ANOVA for 'age' across target 'income' groups: p-value=0.

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
| histogram           | fnlwgt           |                  | Distribution of fnlwgt                | Numeric distribution overview.               |          2 |
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
- charts\hist_fnlwgt.png
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
