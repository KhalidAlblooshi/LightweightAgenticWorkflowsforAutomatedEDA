"""run_all.py — Execute all agent modes on all sample datasets and produce comparison."""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tabulate import tabulate

from src.data_loader import list_sample_datasets, load_csv
from src.profiler import profile_dataset
from src.report_generator import generate_report
from src.evaluator import evaluate_run, save_evaluation_results, generate_comparison_summary
from src.utils import ensure_dir, save_json, sanitize_name


def try_run_mode(mode: str, df, profile, run_output_dir: Path, llm_backend: str = "ollama"):
    """Run a single agent mode, returning (result, runtime) or raising."""
    if mode == "fixed":
        from agents.fixed_pipeline import FixedPipelineAgent
        agent = FixedPipelineAgent(df, profile, run_output_dir)
    elif mode == "rule":
        from agents.rule_based_agent import RuleBasedAgent
        agent = RuleBasedAgent(df, profile, run_output_dir)
    elif mode == "llm":
        from agents.llm_agent import LLMAgent
        agent = LLMAgent(df, profile, run_output_dir, backend=llm_backend)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    start = time.perf_counter()
    result = agent.run()
    return result, time.perf_counter() - start


def main():
    output_root = Path("outputs")
    ensure_dir(output_root)

    datasets = list_sample_datasets()
    if not datasets:
        print("No sample datasets found in data/sample/. Run scripts/download_sample_datasets.py first.")
        sys.exit(1)

    modes = ["fixed", "rule", "llm"]
    all_eval_results: list = []
    summary_rows: list = []

    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            df = load_csv(dataset_path)
            profile = profile_dataset(df, dataset_name)
        except Exception as exc:
            print(f"  ERROR loading {dataset_path}: {exc}")
            continue

        for mode in modes:
            run_output_dir = output_root / sanitize_name(dataset_name) / mode
            ensure_dir(run_output_dir)
            print(f"\n  [{mode.upper()} AGENT]")

            try:
                result, runtime = try_run_mode(mode, df, profile, run_output_dir)
                tool_results = result["tool_results"]
                insights = result["insights"]
                chart_paths = result["chart_paths"]
                tool_log = result["tool_log"]
                recommendations = result["recommendations"]

                save_json(tool_results, run_output_dir / "tool_results.json")
                save_json(tool_log, run_output_dir / "tool_log.json")

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
                all_eval_results.append(eval_result)

                summary_rows.append([
                    dataset_name,
                    mode,
                    f"{runtime:.2f}s",
                    len(tool_log),
                    len(insights),
                    len(chart_paths),
                    eval_result["completeness_score"],
                    eval_result["clarity_score"],
                    eval_result["error_count"],
                ])
                print(f"    Runtime: {runtime:.2f}s  |  Tools: {len(tool_log)}  |  "
                      f"Insights: {len(insights)}  |  Charts: {len(chart_paths)}  |  "
                      f"Completeness: {eval_result['completeness_score']}/10")

            except Exception as exc:
                print(f"    ERROR in {mode} mode: {exc}")
                summary_rows.append([dataset_name, mode, "ERROR", 0, 0, 0, 0, 0, 1])

    # --- Save aggregate outputs ---
    if all_eval_results:
        save_evaluation_results(all_eval_results, output_root)
        generate_comparison_summary(all_eval_results, output_root)

    # --- Print summary table ---
    print(f"\n\n{'='*80}")
    print("  RUN ALL — Summary")
    print(f"{'='*80}")
    headers = [
        "Dataset", "Mode", "Runtime", "Tools",
        "Insights", "Charts", "Completeness", "Clarity", "Errors"
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="rounded_outline"))
    print()


if __name__ == "__main__":
    main()
