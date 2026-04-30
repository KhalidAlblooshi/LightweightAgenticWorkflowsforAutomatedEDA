"""Tool registry — maps tool names to their callable implementations."""

from src.eda_tools import (
    chart_generation,
    categorical_summary,
    correlation_analysis,
    dataset_overview,
    duplicate_analysis,
    insight_generation,
    missing_value_analysis,
    numeric_summary,
    outlier_detection,
    target_aware_analysis,
    visualization_recommendation,
)

TOOL_REGISTRY: dict = {
    "dataset_overview": dataset_overview,
    "missing_value_analysis": missing_value_analysis,
    "duplicate_analysis": duplicate_analysis,
    "numeric_summary": numeric_summary,
    "categorical_summary": categorical_summary,
    "correlation_analysis": correlation_analysis,
    "outlier_detection": outlier_detection,
    "target_aware_analysis": target_aware_analysis,
    "visualization_recommendation": visualization_recommendation,
    "chart_generation": chart_generation,
    "insight_generation": insight_generation,
}


def get_tool(name: str):
    """Return the callable registered under *name*.

    Raises
    ------
    KeyError
        When no tool with the given name is registered.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(
            f"Unknown tool '{name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    return TOOL_REGISTRY[name]


def list_tools() -> list:
    """Return a sorted list of registered tool names."""
    return sorted(TOOL_REGISTRY.keys())
