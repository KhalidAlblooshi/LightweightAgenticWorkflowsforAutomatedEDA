# Statistical Significance Summary

Methods: bootstrap 95% CI for strategy means, Wilcoxon signed-rank pairwise tests, and Friedman tests across strategies.

## Friedman Tests
| metric                          |   n_datasets |   n_strategies |   friedman_statistic |   friedman_p_value |   kendalls_w | kendalls_w_magnitude   |
|---------------------------------|--------------|----------------|----------------------|--------------------|--------------|------------------------|
| runtime_seconds                 |            3 |              3 |              4.66667 |           0.096972 |     0.777778 | large                  |
| number_of_tools_called          |            3 |              3 |              5.63636 |           0.059714 |     0.939394 | large                  |
| number_of_insights_generated    |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| number_of_charts_generated      |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| completeness_score              |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| insight_quality_score           |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| insight_relevance_score         |            3 |              3 |              2       |           0.367879 |     0.333333 | moderate               |
| redundancy_score                |            3 |              3 |              0       |           1        |     0        | negligible             |
| clarity_score                   |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| flesch_reading_ease             |            3 |              3 |              4.66667 |           0.096972 |     0.777778 | large                  |
| readability_score               |            3 |              3 |              4.66667 |           0.096972 |     0.777778 | large                  |
| visualization_suitability_score |            3 |              3 |              0       |           1        |     0        | negligible             |
| efficiency_score                |            3 |              3 |              5.6     |           0.06081  |     0.933333 | large                  |
| overall_quality_score           |            3 |              3 |              6       |           0.049787 |     1        | large                  |
| error_count                     |            3 |              3 |              0       |           1        |     0        | negligible             |

## Pairwise Wilcoxon (All Metrics)
| metric                       | strategy_a   | strategy_b   |   n_datasets |   mean_a |   mean_b |   mean_diff_a_minus_b |   wilcoxon_p_value |   cliffs_delta | cliffs_delta_magnitude   | better_strategy   |   wilcoxon_p_holm | reject_h0_alpha_0_05   |
|------------------------------|--------------|--------------|--------------|----------|----------|-----------------------|--------------------|----------------|--------------------------|-------------------|-------------------|------------------------|
| runtime_seconds              | fixed        | llm          |            3 |   2.1643 |   7.2966 |               -5.1323 |               0.25 |        -1      | large                    | llm               |              0.75 | False                  |
| runtime_seconds              | fixed        | rule         |            3 |   2.1643 |   1.9937 |                0.1706 |               1    |        -0.1111 | negligible               | fixed             |              1    | False                  |
| runtime_seconds              | llm          | rule         |            3 |   7.2966 |   1.9937 |                5.3029 |               0.25 |         1      | large                    | llm               |              0.75 | False                  |
| number_of_tools_called       | fixed        | llm          |            3 |  10      |  10.4333 |               -0.4333 |               0.5  |        -0.6667 | large                    | llm               |              0.75 | False                  |
| number_of_tools_called       | fixed        | rule         |            3 |  10      |  11      |               -1      |               0.25 |        -1      | large                    | rule              |              0.75 | False                  |
| number_of_tools_called       | llm          | rule         |            3 |  10.4333 |  11      |               -0.5667 |               0.25 |        -1      | large                    | rule              |              0.75 | False                  |
| number_of_insights_generated | fixed        | llm          |            3 |   7.6667 |   9.5    |               -1.8333 |               0.25 |        -1      | large                    | llm               |              0.75 | False                  |
| number_of_insights_generated | fixed        | rule         |            3 |   7.6667 |  10.3333 |               -2.6667 |               0.25 |        -1      | large                    | rule              |              0.75 | False                  |
| number_of_insights_generated | llm          | rule         |            3 |   9.5    |  10.3333 |               -0.8333 |               0.25 |        -0.5556 | large                    | rule              |              0.75 | False                  |
| number_of_charts_generated   | fixed        | llm          |            3 |  13.6667 |  12.3667 |                1.3    |               0.25 |         0.5556 | large                    | fixed             |              0.75 | False                  |
| number_of_charts_generated   | fixed        | rule         |            3 |  13.6667 |  13.6667 |                0      |               1    |         0      | negligible               | tie               |              1    | False                  |
| number_of_charts_generated   | llm          | rule         |            3 |  12.3667 |  13.6667 |               -1.3    |               0.25 |        -0.5556 | large                    | rule              |              0.75 | False                  |
| completeness_score           | fixed        | llm          |            3 |  10      |   9.4761 |                0.5239 |               0.25 |         1      | large                    | fixed             |              0.75 | False                  |
| completeness_score           | fixed        | rule         |            3 |  10      |  10      |                0      |               1    |         0      | negligible               | tie               |              1    | False                  |
| completeness_score           | llm          | rule         |            3 |   9.4761 |  10      |               -0.5239 |               0.25 |        -1      | large                    | rule              |              0.75 | False                  |
| insight_quality_score        | fixed        | llm          |            3 |   9.4763 |   9.601  |               -0.1246 |               0.25 |        -0.5556 | large                    | llm               |              0.75 | False                  |
| insight_quality_score        | fixed        | rule         |            3 |   9.4763 |   9.6667 |               -0.1903 |               0.25 |        -0.7778 | large                    | rule              |              0.75 | False                  |
| insight_quality_score        | llm          | rule         |            3 |   9.601  |   9.6667 |               -0.0657 |               0.25 |        -0.5556 | large                    | rule              |              0.75 | False                  |
| insight_relevance_score      | fixed        | llm          |            3 |   9.1667 |   9.45   |               -0.2833 |               1    |        -0.1111 | negligible               | llm               |              1    | False                  |
| insight_relevance_score      | fixed        | rule         |            3 |   9.1667 |   9.3333 |               -0.1667 |               1    |        -0.1111 | negligible               | rule              |              1    | False                  |
| insight_relevance_score      | llm          | rule         |            3 |   9.45   |   9.3333 |                0.1167 |               1    |         0.1111 | negligible               | llm               |              1    | False                  |
| redundancy_score             | fixed        | llm          |            3 |  10      |  10      |                0      |               1    |         0      | negligible               | tie               |              1    | False                  |
| redundancy_score             | fixed        | rule         |            3 |  10      |  10      |                0      |               1    |         0      | negligible               | tie               |              1    | False                  |
| redundancy_score             | llm          | rule         |            3 |  10      |  10      |                0      |               1    |         0      | negligible               | tie               |              1    | False                  |
| clarity_score                | fixed        | llm          |            3 |   9.4763 |   9.3217 |                0.1547 |               0.25 |         0.7778 | large                    | fixed             |              0.75 | False                  |
| clarity_score                | fixed        | rule         |            3 |   9.4763 |   9.291  |                0.1853 |               0.25 |         1      | large                    | fixed             |              0.75 | False                  |
| clarity_score                | llm          | rule         |            3 |   9.3217 |   9.291  |                0.0307 |               0.25 |         0.3333 | medium                   | llm               |              0.75 | False                  |
| flesch_reading_ease          | fixed        | llm          |            3 |  38.4073 |  38.6831 |               -0.2758 |               0.75 |        -0.1111 | negligible               | llm               |              0.75 | False                  |
| flesch_reading_ease          | fixed        | rule         |            3 |  38.4073 |  40.4313 |               -2.024  |               0.25 |        -0.5556 | large                    | rule              |              0.75 | False                  |
| flesch_reading_ease          | llm          | rule         |            3 |  38.6831 |  40.4313 |               -1.7482 |               0.25 |        -0.3333 | medium                   | rule              |              0.75 | False                  |

Full table: comparison_tables/pairwise_significance.csv (45 rows)
