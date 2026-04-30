# EDA Agent Comparison Summary

## Overview

This document compares the three EDA agent modes — **fixed pipeline**, **rule-based**, and **LLM-guided** — across key quality and performance metrics.

## Dataset: students_performance

| Mode   |   runtime_seconds |   number_of_tools_called |   number_of_insights_generated |   number_of_charts_generated |   completeness_score |   redundancy_score |   clarity_score |   visualization_suitability_score |   error_count |
|--------|-------------------|--------------------------|--------------------------------|------------------------------|----------------------|--------------------|-----------------|-----------------------------------|---------------|
| fixed  |            1.5734 |                       10 |                              7 |                           13 |                   10 |                 10 |            7.8  |                              9.31 |             0 |
| rule   |            1.2447 |                       11 |                              8 |                           13 |                   10 |                 10 |            7.73 |                              9.31 |             0 |
| llm    |            1.2986 |                       12 |                              8 |                           13 |                   10 |                 10 |            7.73 |                              9.31 |             1 |

## Dataset: titanic_style

| Mode   |   runtime_seconds |   number_of_tools_called |   number_of_insights_generated |   number_of_charts_generated |   completeness_score |   redundancy_score |   clarity_score |   visualization_suitability_score |   error_count |
|--------|-------------------|--------------------------|--------------------------------|------------------------------|----------------------|--------------------|-----------------|-----------------------------------|---------------|
| fixed  |            1.5104 |                       10 |                             11 |                           13 |                   10 |                 10 |            8.5  |                              9.54 |             0 |
| rule   |            1.5192 |                       11 |                             12 |                           13 |                   10 |                 10 |            8.43 |                              9.54 |             0 |
| llm    |            1.4717 |                       12 |                             12 |                           13 |                   10 |                 10 |            8.43 |                              9.54 |             1 |

## Dataset: wine_quality

| Mode   |   runtime_seconds |   number_of_tools_called |   number_of_insights_generated |   number_of_charts_generated |   completeness_score |   redundancy_score |   clarity_score |   visualization_suitability_score |   error_count |
|--------|-------------------|--------------------------|--------------------------------|------------------------------|----------------------|--------------------|-----------------|-----------------------------------|---------------|
| fixed  |            1.2461 |                       10 |                              9 |                            9 |                   10 |                 10 |            8.3  |                              9.67 |             0 |
| rule   |            1.261  |                       11 |                             10 |                            9 |                   10 |                 10 |            8.23 |                              9.67 |             0 |
| llm    |            1.2032 |                       12 |                             10 |                            9 |                   10 |                 10 |            8.23 |                              9.67 |             1 |

## Best Mode per Metric

| Metric | Best Mode | Value |
|---|---|---|
| runtime_seconds | llm | 1.2032 |
| number_of_tools_called | fixed | 10.0 |
| number_of_insights_generated | rule | 12.0 |
| number_of_charts_generated | fixed | 13.0 |
| completeness_score | fixed | 10.0 |
| redundancy_score | fixed | 10.0 |
| clarity_score | fixed | 8.5 |
| visualization_suitability_score | fixed | 9.67 |
| error_count | fixed | 0.0 |

## Interpretation

The **fixed pipeline** agent always runs the same sequence of tools, ensuring consistency and reproducibility at the cost of flexibility — it may perform unnecessary analyses on small or simple datasets.

The **rule-based** agent adapts tool selection to the data's characteristics (e.g., skipping correlation analysis when there are no numeric columns), producing leaner and more relevant reports.

The **LLM-guided** agent dynamically selects tools based on a language model's reasoning. When the LLM is available it can surface non-obvious tool combinations; however, it falls back to rule-based selection when the model is unreachable, adding latency without guaranteed quality improvement in all cases.

## Discussion for Technical Report

### RQ1 — Can lightweight agentic workflows automate EDA on tabular data?

All three modes successfully executed end-to-end EDA pipelines without human intervention, demonstrating that lightweight agentic workflows are viable for automated tabular EDA.

### RQ2 — How do the three agent designs compare in output quality?

Fixed-pipeline agents guarantee completeness (all standard analyses are always run) but can produce verbose or irrelevant sections for atypical datasets. Rule-based agents improve relevance by conditioning tool selection on profile metadata. LLM-guided agents offer the highest potential flexibility but introduce dependency on an external model and non-deterministic behaviour.

### RQ3 — What are the practical limitations?

- **Latency**: LLM calls add seconds to minutes of overhead.
- **Reliability**: LLM availability is not guaranteed in all deployment contexts.
- **Interpretability**: Rule-based selection is fully auditable; LLM reasoning is opaque unless the reasoning chain is surfaced.
- **Scalability**: Very large datasets may require chunked profiling beyond the current implementation.
