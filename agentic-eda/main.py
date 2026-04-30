"""main.py — CLI entry point for running a single EDA mode on one dataset."""

import argparse
import sys
import time
from pathlib import Path

# Ensure agentic-eda root is on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_csv
from src.profiler import profile_dataset
from src.report_generator import generate_report
from src.evaluator import evaluate_run, save_evaluation_results
from src.utils import ensure_dir, save_json, sanitize_name


def run_agent(mode: str, df, profile, output_dir: Path, llm_backend: str) -> dict:
    """Instantiate and run the requested agent mode."""
    if mode == "fixed":
        from agents.fixed_pipeline import FixedPipelineAgent
        agent = FixedPipelineAgent(df, profile, output_dir)
    elif mode == "rule":
        from agents.rule_based_agent import RuleBasedAgent
        agent = RuleBasedAgent(df, profile, output_dir)
    elif mode == "llm":
        from agents.llm_agent import LLMAgent
        agent = LLMAgent(df, profile, output_dir, backend=llm_backend)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: fixed, rule, llm")
    return agent.run()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic EDA — run automated exploratory data analysis."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "rule", "llm"],
        default="fixed",
        help="Agent mode to use (default: fixed).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for outputs (default: outputs/).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend to use for llm mode (default: ollama).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_root = Path(args.output_dir)
    mode = args.mode
    llm_backend = args.llm_backend

    dataset_name = dataset_path.stem
    run_output_dir = output_root / sanitize_name(dataset_name) / mode
    ensure_dir(run_output_dir)

    print(f"\n{'='*60}")
    print(f"  Agentic EDA")
    print(f"  Dataset : {dataset_path}")
    print(f"  Mode    : {mode}")
    print(f"  Output  : {run_output_dir}")
    print(f"{'='*60}\n")

    # --- Load & profile ---
    print("Loading dataset...")
    df = load_csv(dataset_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded.")

    print("Profiling dataset...")
    profile = profile_dataset(df, dataset_name)
    save_json(profile, run_output_dir / "profile.json")
    print(f"  Numeric: {len(profile['numeric_columns'])} cols, "
          f"Categorical: {len(profile['categorical_columns'])} cols, "
          f"Target: {profile.get('likely_target_col') or 'None'}")

    # --- Run agent ---
    print(f"\nRunning {mode} agent...")
    wall_start = time.perf_counter()
    result = run_agent(mode, df, profile, run_output_dir, llm_backend)
    runtime = time.perf_counter() - wall_start

    tool_results = result["tool_results"]
    insights = result["insights"]
    chart_paths = result["chart_paths"]
    tool_log = result["tool_log"]
    recommendations = result["recommendations"]

    print(f"  Agent finished in {runtime:.2f}s")
    print(f"  Tools called    : {len(tool_log)}")
    print(f"  Insights        : {len(insights)}")
    print(f"  Charts generated: {len(chart_paths)}")

    # --- Save tool results ---
    save_json(tool_results, run_output_dir / "tool_results.json")
    save_json(tool_log, run_output_dir / "tool_log.json")

    # --- Generate report ---
    print("\nGenerating report...")
    generate_report(
        dataset_name=dataset_name,
        mode=mode,
        profile=profile,
        tool_results=tool_results,
        insights=insights,
        chart_paths=chart_paths,
        recommendations=recommendations,
        output_dir=run_output_dir,
    )
    print(f"  Report saved → {run_output_dir / 'report.md'}")

    # --- Evaluate ---
    print("\nEvaluating run...")
    eval_result = evaluate_run(
        dataset_name=dataset_name,
        mode=mode,
        tool_results=tool_results,
        insights=insights,
        chart_paths=chart_paths,
        runtime_seconds=runtime,
        tool_log=tool_log,
        profile=profile,
    )
    save_json(eval_result, run_output_dir / "evaluation.json")
    save_evaluation_results([eval_result], output_root)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("  Evaluation Summary")
    print(f"{'='*60}")
    for k, v in eval_result.items():
        print(f"  {k:<40} {v}")
    print(f"{'='*60}\n")

    if insights:
        print("  Top Insights:")
        for i, ins in enumerate(insights[:5], 1):
            print(f"  {i}. {ins}")
    print()


if __name__ == "__main__":
    main()
