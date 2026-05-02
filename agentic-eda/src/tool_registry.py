"""Safe tool registry for agent execution."""

from __future__ import annotations

from typing import Any, Callable

SAFE_TOOL_NAMES = [
    "dataset_overview",
    "missing_value_analysis",
    "duplicate_analysis",
    "numeric_summary",
    "categorical_summary",
    "correlation_analysis",
    "outlier_detection",
    "target_aware_analysis",
    "visualization_recommendation",
    "chart_generation",
    "insight_generation",
]


class ToolRegistry:
    """Validates and executes only pre-approved tools."""

    def __init__(self, tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        missing = [name for name in SAFE_TOOL_NAMES if name not in tools]
        if missing:
            raise ValueError(f"Missing tool implementations for: {missing}")
        self._tools = tools

    @property
    def allowed_tools(self) -> list[str]:
        return SAFE_TOOL_NAMES.copy()

    def validate(self, tool_names: list[str]) -> list[str]:
        invalid = [name for name in tool_names if name not in SAFE_TOOL_NAMES]
        if invalid:
            raise ValueError(f"Unsupported tools requested: {invalid}")
        return tool_names

    def execute(self, tool_name: str, state: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in SAFE_TOOL_NAMES:
            raise ValueError(f"Tool not allowed: {tool_name}")
        return self._tools[tool_name](state)
