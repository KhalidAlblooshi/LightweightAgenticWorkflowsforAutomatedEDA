"""Generate concise textual insights from tool outputs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from src.utils import safe_div



def _is_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold



def _deduplicate(insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for candidate in insights:
        text = str(candidate.get("insight", "")).strip()
        if not text:
            continue
        if any(_is_similar(text, str(existing.get("insight", ""))) for existing in unique):
            continue
        unique.append(candidate)
    return unique



def generate_insights(tool_results: dict[str, Any]) -> dict[str, Any]:
    """Create insight objects in a report-friendly, JSON-serializable format."""
    raw_insights: list[dict[str, Any]] = []

    overview = tool_results.get("dataset_overview", {})
    if overview:
        raw_insights.append(
            {
                "source_tool": "dataset_overview",
                "insight": (
                    f"Dataset contains {overview.get('row_count', 0)} rows and "
                    f"{overview.get('column_count', 0)} columns."
                ),
            }
        )

    missing = tool_results.get("missing_value_analysis", {})
    if missing:
        missing_cells = missing.get("total_missing_cells", 0)
        missing_pct = missing.get("missing_cell_percentage", 0.0)
        if missing_cells > 0:
            top = missing.get("top_missing_columns", [])
            top_text = ", ".join(f"{x['column']} ({x['missing_count']})" for x in top[:3])
            raw_insights.append(
                {
                    "source_tool": "missing_value_analysis",
                    "insight": (
                        f"Missing data affects {missing_cells} cells ({missing_pct:.2f}%); "
                        f"most impacted columns: {top_text}."
                    ),
                }
            )
        else:
            raw_insights.append(
                {
                    "source_tool": "missing_value_analysis",
                    "insight": "No missing values were detected.",
                }
            )

    dup = tool_results.get("duplicate_analysis", {})
    if dup:
        duplicate_count = dup.get("duplicate_rows", 0)
        duplicate_pct = dup.get("duplicate_percentage", 0.0)
        if duplicate_count > 0:
            raw_insights.append(
                {
                    "source_tool": "duplicate_analysis",
                    "insight": (
                        f"Detected {duplicate_count} duplicate rows ({duplicate_pct:.2f}% of dataset), "
                        "which may bias model evaluation if not handled."
                    ),
                }
            )

    numeric = tool_results.get("numeric_summary", {})
    for row in numeric.get("column_summaries", [])[:3]:
        raw_insights.append(
            {
                "source_tool": "numeric_summary",
                "insight": (
                    f"{row['column']} has mean {row['mean']:.3f}, std {row['std']:.3f}, "
                    f"and spans [{row['min']:.3f}, {row['max']:.3f}]."
                ),
            }
        )

    corr = tool_results.get("correlation_analysis", {})
    strongest = corr.get("strongest_pair")
    if strongest:
        raw_insights.append(
            {
                "source_tool": "correlation_analysis",
                "insight": (
                    f"Strongest absolute correlation is between {strongest.get('feature_1')} and "
                    f"{strongest.get('feature_2')} (r={strongest.get('correlation', 0.0):.3f})."
                ),
            }
        )

    outliers = tool_results.get("outlier_detection", {})
    ranked = outliers.get("ranked_outlier_columns", [])
    if ranked:
        top = ranked[0]
        raw_insights.append(
            {
                "source_tool": "outlier_detection",
                "insight": (
                    f"Column {top['column']} has the highest outlier load "
                    f"({top['outlier_count']} rows, {top['outlier_percentage']:.2f}%)."
                ),
            }
        )

    target = tool_results.get("target_aware_analysis", {})
    if target:
        findings = target.get("findings", [])
        for finding in findings[:3]:
            raw_insights.append(
                {
                    "source_tool": "target_aware_analysis",
                    "insight": finding,
                }
            )

    unique_insights = _deduplicate(raw_insights)

    duplicate_count = max(len(raw_insights) - len(unique_insights), 0)
    redundancy_ratio = safe_div(duplicate_count, max(len(raw_insights), 1), default=0.0)

    for idx, insight in enumerate(unique_insights, start=1):
        insight["insight_id"] = idx

    return {
        "insight_count": len(unique_insights),
        "raw_insight_count": len(raw_insights),
        "redundant_insight_count": duplicate_count,
        "redundancy_ratio": redundancy_ratio,
        "insights": unique_insights,
    }
