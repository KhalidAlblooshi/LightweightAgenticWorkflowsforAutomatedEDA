"""LLM-planned agent with strict tool validation and safe fallback behavior."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from config import SETTINGS
from agents.rule_based_agent import RuleBasedAgent
from src.eda_tools import EDATools, build_tool_map
from src.tool_registry import SAFE_TOOL_NAMES, ToolRegistry


class LLMPlan(BaseModel):
    """Expected strict JSON payload returned by LLM."""

    selected_tools: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    column_guidance: dict[str, list[str]] = Field(default_factory=dict)


class LLMAgent:
    """Selects tools via LLM JSON planning, then executes via safe registry."""

    REQUIRED_POST_TOOLS = [
        "visualization_recommendation",
        "chart_generation",
        "insight_generation",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
        run_dir: Path,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.df = df
        self.profile = profile
        self.run_dir = run_dir
        self.provider = (provider or SETTINGS.llm_provider or "ollama").lower()
        if self.provider == "openai":
            self.model_name = (model_name or SETTINGS.openai_model).strip()
        else:
            self.model_name = (model_name or SETTINGS.ollama_model).strip()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _context_payload(self) -> dict[str, Any]:
        return {
            "dataset_name": self.profile.get("dataset_name"),
            "row_count": self.profile.get("row_count"),
            "column_count": self.profile.get("column_count"),
            "column_types": self.profile.get("column_types", {}),
            "numeric_columns": self.profile.get("numeric_columns", []),
            "categorical_columns": self.profile.get("categorical_columns", []),
            "missing_percentages": self.profile.get("missing_percentages", {}),
            "likely_target_col": self.profile.get("likely_target_col"),
            "sample_rows": self.profile.get("sample_rows", [])[:5],
        }

    def _build_prompt(self) -> str:
        payload = json.dumps(self._context_payload(), indent=2)
        tools_json = json.dumps(SAFE_TOOL_NAMES, indent=2)

        return (
            "You are planning a lightweight EDA workflow for a CSV dataset.\n"
            "Choose ONLY tool names from the allowed list.\n"
            "Return STRICT JSON only, no markdown, no prose before/after JSON.\n"
            "Required JSON schema:\n"
            "{\n"
            '  "selected_tools": ["tool_name"],\n'
            '  "reasoning": "brief explanation",\n'
            '  "column_guidance": {\n'
            '    "focus_columns": ["optional_column_name"],\n'
            '    "avoid_columns": ["optional_column_name"]\n'
            "  }\n"
            "}\n"
            "Include tools that fit dataset properties.\n"
            "Always include visualization_recommendation, chart_generation, and insight_generation.\n"
            f"Allowed tools: {tools_json}\n"
            f"Dataset context: {payload}\n"
        )

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # try to extract first JSON object block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON object.")

        return json.loads(match.group(0))

    def _call_ollama(self, prompt: str) -> str:
        url = f"{SETTINGS.ollama_base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(url, json=payload, timeout=SETTINGS.llm_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _call_openai_compatible(self, prompt: str) -> str:
        if not SETTINGS.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        url = f"{SETTINGS.openai_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {SETTINGS.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=SETTINGS.llm_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI-compatible API returned no choices.")
        return choices[0].get("message", {}).get("content", "")

    def _rule_fallback(self, reason: str) -> dict[str, Any]:
        fallback = RuleBasedAgent(self.df, self.profile, self.run_dir).run()
        fallback["llm_notes"] = f"LLM planning failed. Fallback to rule-based workflow. Reason: {reason}"
        fallback["fallback_used"] = True
        return fallback

    def _execute_selected_tools(
        self,
        selected_tools: list[str],
        llm_notes: str,
        column_guidance: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        tool_impl = EDATools(self.df, self.profile, self.run_dir)
        registry = ToolRegistry(build_tool_map(tool_impl))

        validated = registry.validate(selected_tools)
        column_guidance = column_guidance or {}
        state: dict[str, Any] = {
            "tool_results": {},
            "selected_tools": validated,
            "llm_column_guidance": {
                "focus_columns": column_guidance.get("focus_columns", []),
                "avoid_columns": column_guidance.get("avoid_columns", []),
            },
        }
        tool_log: list[dict[str, Any]] = []

        for step_idx, tool_name in enumerate(validated, start=1):
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
            "selected_tools": validated,
            "tool_results": tool_results,
            "tool_log": tool_log,
            "insights": tool_results.get("insight_generation", {}).get("insights", []),
            "recommendations": tool_results.get("visualization_recommendation", {}).get("recommendations", []),
            "chart_paths": tool_results.get("chart_generation", {}).get("chart_paths", []),
            "llm_notes": llm_notes,
            "fallback_used": False,
        }

    def run(self) -> dict[str, Any]:
        prompt = self._build_prompt()

        try:
            if self.provider == "openai":
                raw_response = self._call_openai_compatible(prompt)
            else:
                raw_response = self._call_ollama(prompt)

            parsed = self._extract_json(raw_response)
            plan = LLMPlan.model_validate(parsed)

            selected = []
            seen = set()
            for name in plan.selected_tools:
                if name in SAFE_TOOL_NAMES and name not in seen:
                    selected.append(name)
                    seen.add(name)
            if not selected:
                raise ValueError("LLM returned empty or invalid tool list.")

            # enforce required post-analysis tools for consistent outputs
            for required_tool in self.REQUIRED_POST_TOOLS:
                if required_tool not in selected:
                    selected.append(required_tool)

            llm_notes = (
                f"Provider: {self.provider}. "
                f"Model: {self.model_name}. "
                f"Reasoning: {plan.reasoning.strip() or 'No reasoning provided.'}"
            )
            dataset_columns = set(self.df.columns.astype(str).tolist())
            raw_guidance = plan.column_guidance or {}
            focus_columns = [c for c in raw_guidance.get("focus_columns", []) if c in dataset_columns]
            avoid_columns = [c for c in raw_guidance.get("avoid_columns", []) if c in dataset_columns]

            return self._execute_selected_tools(
                selected,
                llm_notes,
                column_guidance={
                    "focus_columns": focus_columns,
                    "avoid_columns": avoid_columns,
                },
            )

        except (requests.RequestException, ValidationError, ValueError, json.JSONDecodeError) as exc:
            return self._rule_fallback(str(exc))
