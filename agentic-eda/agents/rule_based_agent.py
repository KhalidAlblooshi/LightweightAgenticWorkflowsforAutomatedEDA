"""Rule-Based Agent — selects tools adaptively based on dataset characteristics."""

import time
from pathlib import Path

import pandas as pd

from src.tool_registry import get_tool


class RuleBasedAgent:
    """Selects and executes EDA tools based on profile-driven rules."""

    def __init__(self, df: pd.DataFrame, profile: dict, output_dir: Path):
        self.df = df
        self.profile = profile
        self.output_dir = Path(output_dir)

    def _select_tools(self) -> list:
        """Apply heuristic rules to produce an ordered tool sequence."""
        tools = []

        # Rule 1: Always run core overview and quality checks
        tools += ["dataset_overview", "missing_value_analysis", "duplicate_analysis"]

        # Rule 2: Numeric analyses if numeric columns are present
        if self.profile["numeric_columns"]:
            tools += ["numeric_summary", "correlation_analysis", "outlier_detection"]

        # Rule 3: Categorical analysis if categorical columns are present
        if self.profile["categorical_columns"]:
            tools.append("categorical_summary")

        # Rule 4: Target-aware analysis if a target column was identified
        if self.profile["has_likely_target"]:
            tools.append("target_aware_analysis")

        # Rule 5: Always close with viz, charts, and insights
        tools += ["visualization_recommendation", "chart_generation", "insight_generation"]

        return tools

    def run(self) -> dict:
        """Execute selected tools and return consolidated results.

        Returns
        -------
        dict with keys: tool_results, insights, chart_paths, tool_log, recommendations
        """
        selected = self._select_tools()
        tool_results: dict = {}
        tool_log: list = []
        chart_paths: list = []
        insights: list = []
        recommendations: list = []

        for tool_name in selected:
            start = time.perf_counter()
            try:
                fn = get_tool(tool_name)

                kwargs: dict = {"mode": "rule", "output_dir": self.output_dir}
                if tool_name == "chart_generation":
                    kwargs["recommendations"] = recommendations
                if tool_name == "insight_generation":
                    kwargs["tool_results"] = tool_results

                result = fn(self.df, self.profile, **kwargs)
                duration = time.perf_counter() - start

                tool_results[tool_name] = result
                tool_log.append({
                    "tool_name": tool_name,
                    "status": "success",
                    "duration_seconds": round(duration, 4),
                    "error_message": None,
                })

                if tool_name == "visualization_recommendation":
                    recommendations = result if isinstance(result, list) else []
                if tool_name == "chart_generation" and isinstance(result, dict):
                    chart_paths = result.get("chart_paths", [])
                if tool_name == "insight_generation" and isinstance(result, dict):
                    insights = result.get("insights", [])

            except Exception as exc:
                duration = time.perf_counter() - start
                tool_log.append({
                    "tool_name": tool_name,
                    "status": "error",
                    "duration_seconds": round(duration, 4),
                    "error_message": str(exc),
                })
                tool_results[tool_name] = {"error": str(exc)}

        return {
            "tool_results": tool_results,
            "insights": insights,
            "chart_paths": chart_paths,
            "tool_log": tool_log,
            "recommendations": recommendations,
        }
