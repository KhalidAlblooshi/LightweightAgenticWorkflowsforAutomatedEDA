"""Fixed Pipeline Agent — always executes the same ordered sequence of tools."""

import time
from pathlib import Path

import pandas as pd

from src.tool_registry import get_tool

_PIPELINE = [
    "dataset_overview",
    "missing_value_analysis",
    "duplicate_analysis",
    "numeric_summary",
    "categorical_summary",
    "correlation_analysis",
    "outlier_detection",
    "visualization_recommendation",
    "chart_generation",
    "insight_generation",
]


class FixedPipelineAgent:
    """Runs a fixed, predetermined sequence of EDA tools."""

    def __init__(self, df: pd.DataFrame, profile: dict, output_dir: Path):
        self.df = df
        self.profile = profile
        self.output_dir = Path(output_dir)

    def run(self) -> dict:
        """Execute the fixed pipeline and return consolidated results.

        Returns
        -------
        dict with keys: tool_results, insights, chart_paths, tool_log, recommendations
        """
        tool_results: dict = {}
        tool_log: list = []
        chart_paths: list = []
        insights: list = []
        recommendations: list = []

        for tool_name in _PIPELINE:
            start = time.perf_counter()
            try:
                fn = get_tool(tool_name)

                kwargs: dict = {"mode": "fixed", "output_dir": self.output_dir}
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

                # Collect recommendations for the chart step
                if tool_name == "visualization_recommendation":
                    recommendations = result if isinstance(result, list) else []
                # Collect chart paths
                if tool_name == "chart_generation" and isinstance(result, dict):
                    chart_paths = result.get("chart_paths", [])
                # Collect insights
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
