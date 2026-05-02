# Automated EDA Report: adult_income

## Dataset Name
adult_income

## Workflow Mode
llm__ollama_qwen2.5_7b

## Selected Tools
- dataset_overview
- missing_value_analysis
- categorical_summary
- correlation_analysis
- target_aware_analysis
- visualization_recommendation
- chart_generation
- insight_generation

## LLM Planning Notes
Provider: ollama. Model: qwen2.5:7b. Reasoning: To understand the dataset and identify patterns related to income, it's essential to start with an overview and handle missing values. Categorical columns are crucial for this classification task, so their summary is necessary. Correlation analysis helps in understanding relationships between numeric features. Target-aware analysis provides insights specific to the income column.

## Dataset Overview
Rows: 32560, Columns: 15, Numeric: 6, Categorical: 9

## Data Quality Findings
Missing cells: 9061 (1.86%).
Duplicate rows: 0 (0.00%).
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
- Strongest absolute correlation is between education_num and hours_per_week (r=0.148).
- Target class distribution: <=50K=24499, >50K=7767.
- ANOVA for 'age' across target 'income' groups: p-value=0.

## Visualization Recommendations
| chart_type   | x            | y             | title                                 | reason                                     |   priority |
|--------------|--------------|---------------|---------------------------------------|--------------------------------------------|------------|
| missing_bar  | column       | missing_count | Missing values by column              | Columns with missing values were detected. |          1 |
| bar          | income       | count         | Category frequencies for income       | Categorical frequency comparison.          |          2 |
| bar          | race         | count         | Category frequencies for race         | Categorical frequency comparison.          |          2 |
| bar          | relationship | count         | Category frequencies for relationship | Categorical frequency comparison.          |          2 |
| bar          | sex          | count         | Category frequencies for sex          | Categorical frequency comparison.          |          2 |

## Chart List
- charts\missing_values_bar.png
- charts\catbar_income.png
- charts\catbar_race.png
- charts\catbar_relationship.png
- charts\catbar_sex.png

## Limitations
- Results depend on heuristic target detection and may not match domain intent.
- Outlier logic is IQR-based and may over-flag skewed distributions.
- LLM mode can fall back to rules when planning output is invalid or unavailable.

## Reproducibility Notes
- Random seed: 42
- Charts generated with matplotlib only (no seaborn).
- Tool execution restricted to safe registry entries.
- Input data source: CSV files under data/sample/.
