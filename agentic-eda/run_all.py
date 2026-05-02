"""Run all workflow modes across all sample datasets and compare outcomes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from agents.fixed_pipeline import FixedPipelineAgent
from agents.llm_agent import LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from config import SETTINGS
from src.data_loader import list_sample_datasets, load_csv
from src.evaluator import evaluate_run, generate_comparison_artifacts, save_evaluation_results
from src.profiler import profile_dataset
from src.report_generator import generate_report
from src.utils import build_llm_label, ensure_dir, reset_directory, sanitize_name, save_json, save_records_csv



def _run_agent(
    mode: str,
    df,
    profile: dict,
    run_dir: Path,
    llm_provider: str,
    llm_model: str | None,
) -> dict:
    if mode == "fixed":
        return FixedPipelineAgent(df, profile, run_dir).run()
    if mode == "rule":
        return RuleBasedAgent(df, profile, run_dir).run()
    if mode == "llm":
        return LLMAgent(df, profile, run_dir, provider=llm_provider, model_name=llm_model).run()
    raise ValueError(f"Unsupported mode: {mode}")


def _record_key(record: dict) -> str:
    return f"{record.get('dataset_name', '')}::{record.get('mode', '')}"


def _normalize_evaluation_record(record: dict) -> dict:
    mode = str(record.get("mode", "") or "")
    strategy = str(record.get("strategy_mode", "") or "")
    llm_label = str(record.get("llm_label", "") or "")

    if strategy.lower() in {"nan", "none", ""}:
        strategy = ""
    if llm_label.lower() in {"nan", "none"}:
        llm_label = ""

    if not strategy:
        if mode.startswith("llm__") or mode == "llm":
            strategy = "llm"
        elif mode in {"fixed", "rule"}:
            strategy = mode
        else:
            strategy = mode

    if not llm_label and mode.startswith("llm__"):
        llm_label = mode.replace("llm__", "", 1)

    record["strategy_mode"] = strategy
    record["llm_label"] = llm_label
    return record


def _load_existing_evaluations(output_root: Path) -> list[dict]:
    eval_path = output_root / "evaluation_results.csv"
    if not eval_path.exists():
        return []
    try:
        df = pd.read_csv(eval_path)
    except Exception:
        return []
    if df.empty:
        return []
    return [_normalize_evaluation_record(row) for row in df.to_dict(orient="records")]



def main() -> None:
    parser = argparse.ArgumentParser(description="Run all modes across all sample datasets.")
    parser.add_argument("--output-dir", default=str(SETTINGS.outputs_dir))
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default=SETTINGS.llm_provider)
    parser.add_argument("--llm-model", default="", help="Optional model override (e.g., qwen2.5:3b).")
    parser.add_argument(
        "--llm-tag",
        default="",
        help="Optional extra tag to differentiate runs with the same model (e.g., promptv2).",
    )
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_dir))
    sample_datasets = list_sample_datasets(SETTINGS.sample_data_dir)
    llm_model = args.llm_model.strip() or None
    llm_tag = args.llm_tag.strip() or None

    if not sample_datasets:
        print("No datasets found in data/sample/. Run scripts/download_sample_datasets.py first.")
        sys.exit(1)

    modes = ["fixed", "rule", "llm"]
    all_evaluations: list[dict] = _load_existing_evaluations(output_root)
    evaluation_map: dict[str, dict] = {_record_key(row): row for row in all_evaluations}

    print("Running all modes on all sample datasets...")
    print(f"Datasets found: {len(sample_datasets)}")

    for dataset_path in sample_datasets:
        dataset_name = sanitize_name(dataset_path.stem)
        print("-" * 72)
        print(f"Dataset: {dataset_name}")

        try:
            df = load_csv(dataset_path)
        except Exception as exc:
            print(f"  Failed to load dataset: {exc}")
            continue

        profile = profile_dataset(df, dataset_name)

        for mode in modes:
            if mode == "llm":
                default_model = SETTINGS.openai_model if args.llm_provider == "openai" else SETTINGS.ollama_model
                model_for_label = llm_model or default_model
                llm_label = build_llm_label(args.llm_provider, model_for_label, tag=llm_tag)
                mode_label = f"llm__{llm_label}"
                run_dir = reset_directory(output_root / dataset_name / "llm" / llm_label)
            else:
                llm_label = ""
                mode_label = mode
                run_dir = reset_directory(output_root / dataset_name / mode)

            save_json(profile, run_dir / "profile.json")

            print(f"  Mode: {mode_label}")
            start = time.perf_counter()

            try:
                run_result = _run_agent(
                    mode,
                    df,
                    profile,
                    run_dir,
                    llm_provider=args.llm_provider,
                    llm_model=llm_model,
                )
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
                evaluation_map[_record_key(evaluation)] = evaluation

                fallback_suffix = " (fallback)" if run_result.get("fallback_used") else ""
                print(
                    "    "
                    f"runtime={evaluation['runtime_seconds']:.3f}s | "
                    f"tools={evaluation['number_of_tools_called']} | "
                    f"insights={evaluation['number_of_insights_generated']} | "
                    f"charts={evaluation['number_of_charts_generated']}" + fallback_suffix
                )

            except Exception as exc:
                print(f"    ERROR: {exc}")
                failure_eval = {
                    "dataset_name": dataset_name,
                    "mode": mode_label,
                    "strategy_mode": mode,
                    "llm_label": llm_label,
                    "runtime_seconds": round(time.perf_counter() - start, 4),
                    "number_of_tools_called": 0,
                    "number_of_insights_generated": 0,
                    "number_of_charts_generated": 0,
                    "completeness_score": 0.0,
                    "insight_quality_score": 0.0,
                    "insight_relevance_score": 0.0,
                    "redundancy_score": 0.0,
                    "clarity_score": 0.0,
                    "flesch_reading_ease": 0.0,
                    "readability_score": 0.0,
                    "visualization_suitability_score": 0.0,
                    "efficiency_score": 0.0,
                    "overall_quality_score": 0.0,
                    "error_count": 1,
                    "fallback_used": False,
                    "quality_warnings": ["run_failed"],
                }
                evaluation_map[_record_key(failure_eval)] = failure_eval

    merged_evaluations = list(evaluation_map.values())
    eval_path = save_evaluation_results(merged_evaluations, output_root)
    generate_comparison_artifacts(merged_evaluations, output_root)

    print("=" * 72)
    print("All runs complete")
    print(f"Evaluation table: {eval_path}")
    print(f"Comparison summary: {output_root / 'comparison_summary.md'}")
    print(f"Comparison tables: {output_root / 'comparison_tables'}")
    print(f"Comparison plots: {output_root / 'comparison_plots'}")
    print("=" * 72)



if __name__ == "__main__":
    main()
