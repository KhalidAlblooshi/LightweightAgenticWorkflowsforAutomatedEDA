"""Fixed pipeline baseline: run a predetermined tool sequence."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.eda_tools import EDATools, build_tool_map
from src.tool_registry import ToolRegistry


class FixedPipelineAgent:
    """Baseline workflow that always runs the same tools in fixed order."""

    FIXED_SEQUENCE = [
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

    def __init__(self, df: pd.DataFrame, profile: dict[str, Any], run_dir: Path) -> None:
        self.df = df
        self.profile = profile
        self.run_dir = run_dir

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def run(self) -> dict[str, Any]:
        tool_impl = EDATools(self.df, self.profile, self.run_dir)
        registry = ToolRegistry(build_tool_map(tool_impl))

        selected_tools = registry.validate(self.FIXED_SEQUENCE.copy())
        state: dict[str, Any] = {
            "tool_results": {},
            "selected_tools": selected_tools,
        }
        tool_log: list[dict[str, Any]] = []

        for step_idx, tool_name in enumerate(selected_tools, start=1):
            start = time.perf_counter()
            started_at = self._timestamp()
            status = "success"
            error_message = ""

            try:
                result = registry.execute(tool_name, state)
                state["tool_results"][tool_name] = result
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                state["tool_results"][tool_name] = {"error": str(exc)}

            duration = time.perf_counter() - start
            tool_log.append(
                {
                    "step": step_idx,
                    "tool_name": tool_name,
                    "status": status,
                    "runtime_seconds": round(duration, 4),
                    "started_at_utc": started_at,
                    "error_message": error_message,
                }
            )

        tool_results = state["tool_results"]
        return {
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "tool_log": tool_log,
            "insights": tool_results.get("insight_generation", {}).get("insights", []),
            "recommendations": tool_results.get("visualization_recommendation", {}).get("recommendations", []),
            "chart_paths": tool_results.get("chart_generation", {}).get("chart_paths", []),
            "llm_notes": "",
        }
