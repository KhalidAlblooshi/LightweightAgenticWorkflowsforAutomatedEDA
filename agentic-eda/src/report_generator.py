"""Report generator — writes a Markdown EDA report."""

from datetime import datetime
from pathlib import Path

from src.utils import ensure_dir


def generate_report(
    dataset_name: str,
    mode: str,
    profile: dict,
    tool_results: dict,
    insights: list,
    chart_paths: list,
    recommendations: list,
    output_dir: Path,
) -> str:
    """Build a Markdown report and write it to *output_dir*/report.md.

    Returns
    -------
    The rendered Markdown string.
    """
    ensure_dir(output_dir)
    lines = []
    a = lines.append

    a(f"# EDA Report: {dataset_name} — {mode} mode")
    a(f"\n_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_\n")

    # -------------------------------------------------------------------
    # Dataset Overview
    # -------------------------------------------------------------------
    a("## Dataset Overview\n")
    a(f"| Property | Value |")
    a(f"|---|---|")
    a(f"| Dataset name | {dataset_name} |")
    a(f"| Agent mode | {mode} |")
    a(f"| Rows | {profile['n_rows']:,} |")
    a(f"| Columns | {profile['n_cols']} |")
    a(f"| Numeric columns | {len(profile['numeric_columns'])} |")
    a(f"| Categorical columns | {len(profile['categorical_columns'])} |")
    a(f"| Datetime columns | {len(profile['datetime_columns'])} |")
    a(f"| Likely target column | {profile.get('likely_target_col') or 'None'} |")

    ov = tool_results.get("dataset_overview", {})
    if ov:
        a(f"| Memory usage (KB) | {ov.get('memory_usage_kb', 'N/A')} |")
    a("")

    if profile["numeric_columns"]:
        a(f"**Numeric columns:** {', '.join(profile['numeric_columns'])}\n")
    if profile["categorical_columns"]:
        a(f"**Categorical columns:** {', '.join(profile['categorical_columns'])}\n")

    # -------------------------------------------------------------------
    # Data Quality
    # -------------------------------------------------------------------
    a("## Data Quality\n")

    mv = tool_results.get("missing_value_analysis", {})
    dup = tool_results.get("duplicate_analysis", {})

    a("### Missing Values\n")
    if mv:
        a(f"- Total missing cells: **{mv.get('total_missing_cells', 0):,}** "
          f"({mv.get('overall_missing_pct', 0):.1f}%)")
        high_miss = mv.get("columns_above_20pct_missing", [])
        if high_miss:
            a(f"- Columns with >20% missing: {', '.join(high_miss)}")
        per_col = mv.get("per_column", {})
        if per_col:
            missing_cols = {c: s for c, s in per_col.items() if s["missing_count"] > 0}
            if missing_cols:
                a("\n| Column | Missing Count | Missing % |")
                a("|---|---|---|")
                for col, s in missing_cols.items():
                    a(f"| {col} | {s['missing_count']} | {s['missing_pct']}% |")
            else:
                a("- No missing values found.")
    else:
        a("_Missing value analysis not performed._")
    a("")

    a("### Duplicates\n")
    if dup:
        a(f"- Duplicate rows: **{dup.get('duplicate_row_count', 0)}** "
          f"({dup.get('duplicate_row_pct', 0):.1f}%)")
    else:
        a("_Duplicate analysis not performed._")
    a("")

    # -------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------
    a("## Statistical Analysis\n")

    num_sum = tool_results.get("numeric_summary", {})
    if num_sum:
        a("### Numeric Summary\n")
        a("| Column | Mean | Median | Std | Min | Max | Skewness |")
        a("|---|---|---|---|---|---|---|")
        for col, s in num_sum.items():
            if isinstance(s, dict) and "mean" in s:
                a(f"| {col} | {s['mean']:.4f} | {s['median']:.4f} | "
                  f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                  f"{s.get('skewness', 'N/A')} |")
        a("")

    cat_sum = tool_results.get("categorical_summary", {})
    if cat_sum:
        a("### Categorical Summary\n")
        for col, s in cat_sum.items():
            if isinstance(s, dict) and "unique_count" in s:
                a(f"**{col}** — {s['unique_count']} unique values, "
                  f"mode: `{s.get('mode', 'N/A')}`\n")
                top = s.get("top_value_counts", {})
                if top:
                    a("| Value | Count |")
                    a("|---|---|")
                    for val, cnt in list(top.items())[:5]:
                        a(f"| {val} | {cnt} |")
                a("")

    corr = tool_results.get("correlation_analysis", {})
    if corr and not corr.get("skipped"):
        a("### Top Correlated Pairs\n")
        pairs = corr.get("top_correlated_pairs", [])
        if pairs:
            a("| Column 1 | Column 2 | Pearson r |")
            a("|---|---|---|")
            for p in pairs:
                a(f"| {p['col1']} | {p['col2']} | {p['correlation']:.4f} |")
        a("")

    out_res = tool_results.get("outlier_detection", {})
    if out_res:
        a("### Outlier Detection (IQR method)\n")
        a("| Column | Outlier Count | Outlier % |")
        a("|---|---|---|")
        for col, s in out_res.items():
            if isinstance(s, dict) and "outlier_count" in s:
                a(f"| {col} | {s['outlier_count']} | {s['outlier_pct']}% |")
        a("")

    ta = tool_results.get("target_aware_analysis", {})
    if ta and not ta.get("skipped"):
        a("### Target-Aware Analysis\n")
        a(f"- Target column: **{ta.get('target_col', 'N/A')}**")
        a(f"- Target type: {ta.get('target_type', 'N/A')}")
        if ta.get("class_distribution"):
            a("\n**Class distribution:**\n")
            a("| Class | % |")
            a("|---|---|")
            for cls, pct in ta["class_distribution"].items():
                a(f"| {cls} | {pct}% |")
        a("")

    # -------------------------------------------------------------------
    # Key Insights
    # -------------------------------------------------------------------
    a("## Key Insights\n")
    if insights:
        for i, insight in enumerate(insights, 1):
            a(f"{i}. {insight}")
    else:
        a("_No insights generated._")
    a("")

    # -------------------------------------------------------------------
    # Visualization Recommendations
    # -------------------------------------------------------------------
    a("## Visualization Recommendations\n")
    if recommendations:
        a("| # | Chart Type | Columns | Rationale |")
        a("|---|---|---|---|")
        for i, rec in enumerate(recommendations, 1):
            cols_str = ", ".join(rec.get("columns", []))
            a(f"| {i} | {rec.get('chart_type', '')} | {cols_str} | "
              f"{rec.get('rationale', '')} |")
    else:
        a("_No visualizations recommended._")
    a("")

    # -------------------------------------------------------------------
    # Generated Charts
    # -------------------------------------------------------------------
    a("## Generated Charts\n")
    if chart_paths:
        for path in chart_paths:
            fname = Path(path).name
            a(f"- `{fname}`")
    else:
        a("_No charts generated._")
    a("")

    # -------------------------------------------------------------------
    # Limitations
    # -------------------------------------------------------------------
    a("## Limitations\n")
    a(
        "- This report was generated automatically and may not capture all nuances of the data.\n"
        "- Statistical tests (normality, significance) are not performed.\n"
        "- Insights are heuristic and should be validated by a domain expert.\n"
        "- LLM-based tool selection may vary across runs.\n"
        "- Visualizations are limited to the most informative subset of columns."
    )
    a("")

    # -------------------------------------------------------------------
    # Reproducibility Notes
    # -------------------------------------------------------------------
    a("## Reproducibility Notes\n")
    a(
        "- Random seed: 42\n"
        "- All synthetic data generated with `numpy.random.seed(42)`.\n"
        f"- Agent mode: **{mode}**\n"
        "- To reproduce: `python main.py --dataset <path> --mode " + mode + "`"
    )
    a("")

    markdown = "\n".join(lines)

    report_path = output_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    return markdown
