"""Run one workflow mode on a single CSV dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agents.fixed_pipeline import FixedPipelineAgent
from agents.llm_agent import LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from config import SETTINGS
from src.data_loader import load_csv
from src.evaluator import evaluate_run
from src.profiler import profile_dataset
from src.report_generator import generate_report
from src.utils import build_llm_label, reset_directory, sanitize_name, save_json, save_records_csv



def _run_mode(
    mode: str,
    df,
    profile: dict,
    run_dir: Path,
    llm_provider: str,
    llm_model: str | None,
) -> dict:
    if mode == "fixed":
        agent = FixedPipelineAgent(df, profile, run_dir)
    elif mode == "rule":
        agent = RuleBasedAgent(df, profile, run_dir)
    elif mode == "llm":
        agent = LLMAgent(df, profile, run_dir, provider=llm_provider, model_name=llm_model)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return agent.run()



def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated EDA for one dataset.")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset.")
    parser.add_argument("--mode", choices=["fixed", "rule", "llm"], default="fixed")
    parser.add_argument("--output-dir", default=str(SETTINGS.outputs_dir))
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default=SETTINGS.llm_provider)
    parser.add_argument("--llm-model", default="", help="Optional model override (e.g., qwen2.5:3b).")
    parser.add_argument(
        "--llm-tag",
        default="",
        help="Optional extra tag to differentiate runs with the same model (e.g., promptv2).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_root = Path(args.output_dir)
    mode = args.mode
    llm_model = args.llm_model.strip() or None

    df = load_csv(dataset_path)
    dataset_name = sanitize_name(dataset_path.stem)

    if mode == "llm":
        default_model = SETTINGS.openai_model if args.llm_provider == "openai" else SETTINGS.ollama_model
        model_for_label = llm_model or default_model
        llm_label = build_llm_label(args.llm_provider, model_for_label, tag=args.llm_tag.strip() or None)
        mode_label = f"llm__{llm_label}"
        run_dir = reset_directory(output_root / dataset_name / "llm" / llm_label)
    else:
        llm_label = ""
        mode_label = mode
        run_dir = reset_directory(output_root / dataset_name / mode)

    profile = profile_dataset(df, dataset_name)
    save_json(profile, run_dir / "profile.json")

    start = time.perf_counter()
    run_result = _run_mode(mode, df, profile, run_dir, args.llm_provider, llm_model)
    runtime_seconds = time.perf_counter() - start

    selected_tools = run_result.get("selected_tools", [])
    tool_results = run_result.get("tool_results", {})
    tool_log = run_result.get("tool_log", [])
    insights = run_result.get("insights", [])
    recommendations = run_result.get("recommendations", [])

    save_json(tool_results, run_dir / "tool_results.json")
    save_records_csv(tool_log, run_dir / "tool_log.csv")
    save_records_csv(insights, run_dir / "insights.csv")
    save_records_csv(recommendations, run_dir / "visualization_recommendations.csv")

    report_text = generate_report(
        dataset_name=dataset_name,
        mode=mode_label,
        selected_tools=selected_tools,
        profile=profile,
        tool_results=tool_results,
        output_dir=run_dir,
        llm_notes=run_result.get("llm_notes"),
    )

    evaluation = evaluate_run(
        dataset_name=dataset_name,
        mode=mode_label,
        strategy_mode=mode,
        llm_label=llm_label,
        selected_tools=selected_tools,
        tool_results=tool_results,
        tool_log=tool_log,
        profile=profile,
        runtime_seconds=runtime_seconds,
        report_text=report_text,
        fallback_used=bool(run_result.get("fallback_used")),
    )
    save_json(evaluation, run_dir / "evaluation.json")

    print("=" * 72)
    print("Agentic EDA run complete")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Mode: {mode_label}")
    print(f"Output folder: {run_dir}")
    print(f"Runtime (s): {evaluation['runtime_seconds']}")
    print(f"Tools called: {evaluation['number_of_tools_called']}")
    print(f"Insights: {evaluation['number_of_insights_generated']}")
    print(f"Charts: {evaluation['number_of_charts_generated']}")
    print(f"Completeness score: {evaluation['completeness_score']}")
    print(f"Error count: {evaluation['error_count']}")
    if run_result.get("fallback_used"):
        print("LLM fallback: rule-based plan was used due to planning failure.")
    print("=" * 72)



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        print(f"ERROR: {exc}")
        sys.exit(1)
