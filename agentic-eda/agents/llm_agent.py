"""LLM Agent — uses an LLM to select tools, with rule-based fallback."""

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import config
from src.tool_registry import get_tool, list_tools
from agents.rule_based_agent import RuleBasedAgent

_DEPENDENCY_ORDER = [
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


class LLMAgent:
    """Selects EDA tools using an LLM, falling back to rule-based selection."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: dict,
        output_dir: Path,
        backend: str = "ollama",
    ):
        self.df = df
        self.profile = profile
        self.output_dir = Path(output_dir)
        self.backend = backend
        self._fallback_used = False

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, profile: dict) -> str:
        allowed = list_tools()
        schema_lines = []
        for col, dtype in profile["dtypes"].items():
            missing = profile["missing_pct"].get(col, 0)
            schema_lines.append(f"  - {col}: {dtype} (missing: {missing:.1f}%)")

        sample_str = json.dumps(profile["sample_rows"][:3], indent=2)

        return f"""You are an expert data analyst. Your task is to select the most appropriate
EDA (Exploratory Data Analysis) tools for the dataset described below.

## Dataset: {profile['dataset_name']}
- Rows: {profile['n_rows']}
- Columns: {profile['n_cols']}
- Numeric columns: {profile['numeric_columns']}
- Categorical columns: {profile['categorical_columns']}
- Likely target column: {profile.get('likely_target_col') or 'None'}

## Column Schema:
{chr(10).join(schema_lines)}

## Sample Rows (first 3):
{sample_str}

## Available Tools:
{json.dumps(allowed, indent=2)}

## Instructions:
Select the most relevant tools for this dataset. Always include:
- dataset_overview, missing_value_analysis, duplicate_analysis
- visualization_recommendation, chart_generation, insight_generation

Additionally select tools appropriate to the data characteristics.

Respond with ONLY a valid JSON object in exactly this format:
{{
  "selected_tools": ["tool1", "tool2", ...],
  "reasoning": "Brief explanation of tool selection"
}}

Do not include any text outside the JSON object."""

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> Optional[str]:
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def _call_openai(self, prompt: str) -> Optional[str]:
        url = f"{config.OPENAI_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.backend == "openai":
            return self._call_openai(prompt)
        return self._call_ollama(prompt)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> list:
        """Extract selected_tools list from LLM JSON response."""
        if not response:
            return []
        # Try to extract JSON from the response
        try:
            data = json.loads(response.strip())
            tools = data.get("selected_tools", [])
            return [t for t in tools if isinstance(t, str)]
        except json.JSONDecodeError:
            pass
        # Try to find JSON block inside the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(response[start:end])
                tools = data.get("selected_tools", [])
                return [t for t in tools if isinstance(t, str)]
            except json.JSONDecodeError:
                pass
        return []

    def _validate_tools(self, tools: list) -> list:
        """Remove unknown tool names and ensure mandatory tools are included."""
        valid_tools = set(list_tools())
        filtered = [t for t in tools if t in valid_tools]

        mandatory = ["dataset_overview", "missing_value_analysis", "duplicate_analysis",
                     "visualization_recommendation", "chart_generation", "insight_generation"]
        for t in mandatory:
            if t not in filtered:
                filtered.append(t)
        return filtered

    def _sort_by_dependency(self, tools: list) -> list:
        """Reorder tool names to respect execution dependency order."""
        ordered = []
        for t in _DEPENDENCY_ORDER:
            if t in tools:
                ordered.append(t)
        # Append any leftover tools not in the dependency list
        for t in tools:
            if t not in ordered:
                ordered.append(t)
        return ordered

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Select and execute tools, returning consolidated results.

        Returns
        -------
        dict with keys: tool_results, insights, chart_paths, tool_log, recommendations
        """
        # Step 1: Try LLM tool selection
        selected_tools: list = []
        llm_log_entry: dict = {}
        llm_start = time.perf_counter()

        try:
            prompt = self._build_prompt(self.profile)
            response = self._call_llm(prompt)
            raw_tools = self._parse_response(response)
            if not raw_tools:
                raise ValueError("LLM returned no valid tool selection.")
            selected_tools = self._sort_by_dependency(self._validate_tools(raw_tools))
            llm_log_entry = {
                "tool_name": "_llm_selection",
                "status": "success",
                "duration_seconds": round(time.perf_counter() - llm_start, 4),
                "error_message": None,
            }
            print(f"  [LLMAgent] Tools selected by LLM: {selected_tools}")
        except Exception as exc:
            self._fallback_used = True
            llm_log_entry = {
                "tool_name": "_llm_selection",
                "status": "error",
                "duration_seconds": round(time.perf_counter() - llm_start, 4),
                "error_message": f"LLM unavailable/failed: {exc}. Using rule-based fallback.",
            }
            print(f"  [LLMAgent] LLM unavailable ({exc}). Falling back to rule-based agent.")
            fallback = RuleBasedAgent(self.df, self.profile, self.output_dir)
            result = fallback.run()
            result["tool_log"].insert(0, llm_log_entry)
            return result

        # Step 2: Execute selected tools
        tool_results: dict = {}
        tool_log: list = [llm_log_entry]
        chart_paths: list = []
        insights: list = []
        recommendations: list = []

        for tool_name in selected_tools:
            start = time.perf_counter()
            try:
                fn = get_tool(tool_name)

                kwargs: dict = {"mode": "llm", "output_dir": self.output_dir}
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
