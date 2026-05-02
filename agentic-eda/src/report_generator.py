"""Markdown report generation for each dataset/mode execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabulate import tabulate

from config import SETTINGS
from src.utils import write_text



def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No data available."
    table_rows = [[row.get(col, "") for col in columns] for row in rows]
    return tabulate(table_rows, headers=columns, tablefmt="github")



def generate_report(
    dataset_name: str,
    mode: str,
    selected_tools: list[str],
    profile: dict[str, Any],
    tool_results: dict[str, Any],
    output_dir: Path,
    llm_notes: str | None = None,
) -> str:
    """Create required report.md content and write it to output directory."""
    overview = tool_results.get("dataset_overview", {})
    missing = tool_results.get("missing_value_analysis", {})
    dup = tool_results.get("duplicate_analysis", {})
    numeric = tool_results.get("numeric_summary", {})
    categorical = tool_results.get("categorical_summary", {})
    corr = tool_results.get("correlation_analysis", {})
    outlier = tool_results.get("outlier_detection", {})
    target = tool_results.get("target_aware_analysis", {})
    viz = tool_results.get("visualization_recommendation", {})
    chart_gen = tool_results.get("chart_generation", {})
    insight_payload = tool_results.get("insight_generation", {})

    insights = insight_payload.get("insights", [])
    recommendations = viz.get("recommendations", [])
    chart_paths = chart_gen.get("chart_paths", [])

    top_numeric = numeric.get("column_summaries", [])[:6]
    top_categorical = categorical.get("column_summaries", [])[:6]
    top_outliers = outlier.get("ranked_outlier_columns", [])[:6]

    lines: list[str] = []
    lines.append(f"# Automated EDA Report: {dataset_name}")
    lines.append("")
    lines.append("## Dataset Name")
    lines.append(dataset_name)
    lines.append("")
    lines.append("## Workflow Mode")
    lines.append(mode)
    lines.append("")
    lines.append("## Selected Tools")
    lines.append("\n".join(f"- {tool}" for tool in selected_tools) or "- None")
    lines.append("")

    if llm_notes:
        lines.append("## LLM Planning Notes")
        lines.append(llm_notes)
        lines.append("")

    lines.append("## Dataset Overview")
    lines.append(
        f"Rows: {overview.get('row_count', profile.get('row_count', 0))}, "
        f"Columns: {overview.get('column_count', profile.get('column_count', 0))}, "
        f"Numeric: {overview.get('numeric_column_count', len(profile.get('numeric_columns', [])))}, "
        f"Categorical: {overview.get('categorical_column_count', len(profile.get('categorical_columns', [])))}"
    )
    lines.append("")

    lines.append("## Data Quality Findings")
    lines.append(
        f"Missing cells: {missing.get('total_missing_cells', 0)} "
        f"({missing.get('missing_cell_percentage', 0.0):.2f}%)."
    )
    lines.append(
        f"Duplicate rows: {dup.get('duplicate_rows', 0)} "
        f"({dup.get('duplicate_percentage', 0.0):.2f}%)."
    )
    if missing.get("top_missing_columns"):
        lines.append("Top missing columns:")
        lines.append(
            _markdown_table(
                missing.get("top_missing_columns", [])[:10],
                ["column", "missing_count", "missing_percentage"],
            )
        )
    lines.append("")

    lines.append("## Statistical Findings")
    if top_numeric:
        lines.append("### Numeric Summary")
        lines.append(
            _markdown_table(
                top_numeric,
                ["column", "mean", "std", "min", "q1", "median", "q3", "max", "skewness"],
            )
        )
    if top_categorical:
        lines.append("### Categorical Summary")
        cat_rows: list[dict[str, Any]] = []
        for row in top_categorical:
            first = row.get("top_categories", [])[:3]
            cat_rows.append(
                {
                    "column": row.get("column"),
                    "unique_count": row.get("unique_count"),
                    "top_categories": ", ".join(
                        f"{x['category']} ({x['count']})" for x in first
                    ),
                }
            )
        lines.append(_markdown_table(cat_rows, ["column", "unique_count", "top_categories"]))

    if corr.get("strongest_pair"):
        pair = corr["strongest_pair"]
        lines.append("### Correlation Highlights")
        lines.append(
            f"Strongest pair: {pair['feature_1']} vs {pair['feature_2']} "
            f"(r={pair['correlation']:.3f})."
        )

    if top_outliers:
        lines.append("### Outlier Detection")
        lines.append(
            _markdown_table(
                top_outliers,
                ["column", "outlier_count", "outlier_percentage", "lower_bound", "upper_bound"],
            )
        )

    if target and target.get("analysis_type") != "none":
        lines.append("### Target-Aware Analysis")
        lines.append(f"Target column: {target.get('target_column')}")
        for finding in target.get("findings", [])[:8]:
            lines.append(f"- {finding}")
    lines.append("")

    lines.append("## Key Insights")
    if insights:
        for row in insights:
            lines.append(f"- {row.get('insight')}")
    else:
        lines.append("No insights were generated.")
    lines.append("")

    lines.append("## Visualization Recommendations")
    if recommendations:
        lines.append(
            _markdown_table(
                recommendations,
                ["chart_type", "x", "y", "title", "reason", "priority"],
            )
        )
    else:
        lines.append("No visualization recommendations available.")
    lines.append("")

    lines.append("## Chart List")
    if chart_paths:
        for chart in chart_paths:
            lines.append(f"- {chart}")
    else:
        lines.append("No charts were generated.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- Results depend on heuristic target detection and may not match domain intent.")
    lines.append("- Outlier logic is IQR-based and may over-flag skewed distributions.")
    lines.append("- LLM mode can fall back to rules when planning output is invalid or unavailable.")
    lines.append("")

    lines.append("## Reproducibility Notes")
    lines.append(f"- Random seed: {SETTINGS.random_seed}")
    lines.append("- Charts generated with matplotlib only (no seaborn).")
    lines.append("- Tool execution restricted to safe registry entries.")
    lines.append("- Input data source: CSV files under data/sample/.")

    report_text = "\n".join(lines) + "\n"
    write_text(output_dir / "report.md", report_text)
    return report_text
