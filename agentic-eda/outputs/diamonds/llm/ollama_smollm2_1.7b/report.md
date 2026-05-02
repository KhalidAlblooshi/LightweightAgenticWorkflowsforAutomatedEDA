# Automated EDA Report: diamonds

## Dataset Name
diamonds

## Workflow Mode
llm__ollama_smollm2_1.7b

## Selected Tools
- dataset_overview
- missing_value_analysis
- duplicate_analysis
- numeric_summary
- categorical_summary
- correlation_analysis
- outlier_detection
- target_aware_analysis
- visualization_recommendation
- chart_generation
- insight_generation

## LLM Planning Notes
Provider: ollama. Model: smollm2:1.7b. Reasoning: This EDA workflow will provide an overview of the dataset, identify missing values and duplicates, summarize numeric and categorical columns, analyze correlations between variables, detect outliers, understand target distribution, recommend visualizations based on insights, generate charts for visualization recommendations, and offer actionable insights.

## Dataset Overview
Rows: 53940, Columns: 10, Numeric: 7, Categorical: 3

## Data Quality Findings
Missing cells: 2690 (0.50%).
Duplicate rows: 132 (0.24%).
Top missing columns:
| column   |   missing_count |   missing_percentage |
|----------|-----------------|----------------------|
| cut      |             287 |             0.532073 |
| depth    |             284 |             0.526511 |
| y        |             281 |             0.520949 |
| z        |             280 |             0.519095 |
| table    |             270 |             0.500556 |
| x        |             265 |             0.491287 |
| carat    |             261 |             0.483871 |
| color    |             261 |             0.483871 |
| clarity  |             257 |             0.476455 |
| price    |             244 |             0.452354 |

## Statistical Findings
### Numeric Summary
| column   |        mean |         std |   min |     q1 |   median |      q3 |      max |   skewness |
|----------|-------------|-------------|-------|--------|----------|---------|----------|------------|
| carat    |    0.797939 |    0.473942 |   0.2 |   0.4  |     0.7  |    1.04 |     5.01 |  1.11646   |
| depth    |   61.75     |    1.43266  |  43   |  61    |    61.8  |   62.5  |    79    | -0.0794444 |
| table    |   57.4578   |    2.23478  |  43   |  56    |    57    |   59    |    95    |  0.797025  |
| price    | 3933.26     | 3989.47     | 326   | 951    |  2401    | 5324    | 18823    |  1.61786   |
| x        |    5.73129  |    1.12191  |   0   |   4.71 |     5.7  |    6.54 |    10.74 |  0.378162  |
| y        |    5.735    |    1.14241  |   0   |   4.72 |     5.71 |    6.54 |    58.9  |  2.44294   |
### Categorical Summary
| column   |   unique_count | top_categories                                    |
|----------|----------------|---------------------------------------------------|
| cut      |              6 | Ideal (21444), Premium (13723), Very Good (12005) |
| color    |              8 | G (11239), E (9751), F (9496)                     |
| clarity  |              9 | SI1 (13009), VS2 (12191), SI2 (9152)              |
### Correlation Highlights
Strongest pair: carat vs x (r=0.975).
### Outlier Detection
| column   |   outlier_count |   outlier_percentage |   lower_bound |   upper_bound |
|----------|-----------------|----------------------|---------------|---------------|
| price    |            3523 |            6.56101   |     -5608.5   |     11883.5   |
| depth    |            2530 |            4.71522   |        58.75  |        64.75  |
| carat    |            1874 |            3.49112   |        -0.56  |         2     |
| table    |             602 |            1.12167   |        51.5   |        63.5   |
| z        |              49 |            0.0913157 |         1.215 |         5.735 |
| x        |              32 |            0.0596181 |         1.965 |         9.285 |
### Target-Aware Analysis
Target column: price
- Target 'price' correlation with 'carat' is 0.922.
- Target 'price' correlation with 'depth' is -0.010.
- Target 'price' correlation with 'table' is 0.127.
- Target 'price' correlation with 'x' is 0.884.
- Target 'price' correlation with 'y' is 0.865.
- Target 'price' correlation with 'z' is 0.861.

## Key Insights
- Dataset contains 53940 rows and 10 columns.
- Missing data affects 2690 cells (0.50%); most impacted columns: cut (287), depth (284), y (281).
- Detected 132 duplicate rows (0.24% of dataset), which may bias model evaluation if not handled.
- carat has mean 0.798, std 0.474, and spans [0.200, 5.010].
- depth has mean 61.750, std 1.433, and spans [43.000, 79.000].
- table has mean 57.458, std 2.235, and spans [43.000, 95.000].
- Strongest absolute correlation is between carat and x (r=0.975).
- Column price has the highest outlier load (3523 rows, 6.56%).
- Target 'price' correlation with 'carat' is 0.922.
- Target 'price' correlation with 'depth' is -0.010.
- Target 'price' correlation with 'table' is 0.127.

## Visualization Recommendations
| chart_type          | x                | y                | title                         | reason                                       |   priority |
|---------------------|------------------|------------------|-------------------------------|----------------------------------------------|------------|
| correlation_heatmap | numeric_features | numeric_features | Correlation heatmap           | Multiple numeric columns are available.      |          1 |
| missing_bar         | column           | missing_count    | Missing values by column      | Columns with missing values were detected.   |          1 |
| scatter             | carat            | x                | Strongest numeric correlation | Useful for validating linear trend strength. |          1 |
| histogram           | carat            |                  | Distribution of carat         | Numeric distribution overview.               |          2 |
| histogram           | depth            |                  | Distribution of depth         | Numeric distribution overview.               |          2 |
| histogram           | price            |                  | Distribution of price         | Numeric distribution overview.               |          2 |
| histogram           | table            |                  | Distribution of table         | Numeric distribution overview.               |          2 |
| histogram           | y                |                  | Distribution of y             | Numeric distribution overview.               |          2 |
| histogram           | z                |                  | Distribution of z             | Numeric distribution overview.               |          2 |

## Chart List
- charts\correlation_heatmap.png
- charts\missing_values_bar.png
- charts\scatter_carat_vs_x.png
- charts\hist_carat.png
- charts\hist_depth.png
- charts\hist_price.png
- charts\hist_table.png
- charts\hist_y.png
- charts\hist_z.png

## Limitations
- Results depend on heuristic target detection and may not match domain intent.
- Outlier logic is IQR-based and may over-flag skewed distributions.
- LLM mode can fall back to rules when planning output is invalid or unavailable.

## Reproducibility Notes
- Random seed: 42
- Charts generated with matplotlib only (no seaborn).
- Tool execution restricted to safe registry entries.
- Input data source: CSV files under data/sample/.
