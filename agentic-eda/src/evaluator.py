"""Evaluation metrics and cross-run comparison artifacts."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from src.utils import ensure_dir, save_records_csv, safe_div, write_text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _has_meaningful_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        if not value:
            return False
        non_error_keys = [k for k in value.keys() if k != "error"]
        if not non_error_keys:
            return False
        for key in non_error_keys:
            val = value.get(key)
            if isinstance(val, (list, dict, str)) and _has_meaningful_content(val):
                return True
            if isinstance(val, (int, float)):
                return True
        return False
    return True


def _column_stats(profile: dict[str, Any], col: str) -> dict[str, Any]:
    return profile.get("column_stats", {}).get(col, {})


def _is_identifier_like(profile: dict[str, Any], col: str) -> bool:
    if col in set(profile.get("id_like_columns", [])):
        return True
    return bool(_column_stats(profile, col).get("is_identifier_like", False))


def _estimate_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 1
    vowels = "aeiouy"
    groups = 0
    prev_is_vowel = False
    for ch in cleaned:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            groups += 1
        prev_is_vowel = is_vowel
    if cleaned.endswith("e") and groups > 1:
        groups -= 1
    return max(groups, 1)


def _flesch_reading_ease(text: str) -> float:
    """Approximate Flesch Reading Ease (Flesch, 1948)."""
    if not text or not text.strip():
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sentences or not words:
        return 0.0

    avg_sentence_length = len(words) / max(len(sentences), 1)
    avg_syllables_per_word = sum(_estimate_syllables(word) for word in words) / max(len(words), 1)
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return float(score)


def _flesch_to_ten_point(flesch_score: float) -> float:
    capped = min(100.0, max(0.0, flesch_score))
    return round((capped / 100.0) * 10.0, 3)


def _completeness_score(tool_results: dict[str, Any], profile: dict[str, Any]) -> float:
    numeric_cols = profile.get("numeric_columns", [])
    categorical_cols = profile.get("categorical_columns", [])

    numeric_summary = tool_results.get("numeric_summary", {})
    categorical_summary = tool_results.get("categorical_summary", {})
    correlation = tool_results.get("correlation_analysis", {})
    outliers = tool_results.get("outlier_detection", {})
    insight_payload = tool_results.get("insight_generation", {})
    recommendations = tool_results.get("visualization_recommendation", {}).get("recommendations", [])
    chart_info = tool_results.get("chart_generation", {})
    chart_paths = chart_info.get("chart_paths", [])
    plot_failures = chart_info.get("plot_failures", [])

    checks: list[bool] = []
    checks.append(
        _has_meaningful_content(tool_results.get("missing_value_analysis"))
        and _has_meaningful_content(tool_results.get("duplicate_analysis"))
    )
    if numeric_cols:
        checks.append(bool(numeric_summary.get("column_summaries")))
        checks.append(bool(correlation.get("top_pairs") or correlation.get("strongest_pair")))
        checks.append(bool(outliers.get("ranked_outlier_columns")))
    if categorical_cols:
        checks.append(bool(categorical_summary.get("column_summaries")))
    checks.append(bool(insight_payload.get("insights")))

    viz_ok = bool(recommendations) and bool(chart_paths)
    if recommendations:
        coverage = safe_div(len(chart_paths), len(recommendations), default=0.0)
        viz_ok = viz_ok and coverage >= 0.6
    viz_ok = viz_ok and len(plot_failures) == 0
    checks.append(viz_ok)

    if not checks:
        return 0.0
    return round((sum(1 for c in checks if c) / len(checks)) * 10.0, 3)


def _redundancy_score(insights: list[dict[str, Any]]) -> float:
    if not insights:
        return 0.0

    texts = [str(x.get("insight", "")).strip() for x in insights if str(x.get("insight", "")).strip()]
    if len(texts) < 2:
        return 10.0

    pair_count = 0
    near_duplicates = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pair_count += 1
            if _similarity(texts[i], texts[j]) >= 0.88:
                near_duplicates += 1

    dup_ratio = safe_div(near_duplicates, pair_count, default=0.0)
    return round(max(0.0, 10.0 * (1.0 - dup_ratio)), 3)


def _insight_quality_score(insights: list[dict[str, Any]]) -> float:
    """Score insight quality using evidence density, source diversity, and readability."""
    if not insights:
        return 0.0

    texts = [str(row.get("insight", "")).strip() for row in insights if str(row.get("insight", "")).strip()]
    if not texts:
        return 0.0

    numeric_pattern = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?%?")
    evidence_ratio = safe_div(sum(1 for text in texts if numeric_pattern.search(text)), len(texts), default=0.0)
    readable_ratio = safe_div(sum(1 for text in texts if 40 <= len(text) <= 240), len(texts), default=0.0)
    source_diversity = safe_div(
        len({str(row.get("source_tool", "")).strip() for row in insights if row.get("source_tool")}),
        min(len(texts), 8),
        default=0.0,
    )

    score = (0.45 * evidence_ratio + 0.35 * readable_ratio + 0.20 * min(1.0, source_diversity)) * 10.0
    return round(score, 3)


def _insight_relevance_score(
    insights: list[dict[str, Any]],
    profile: dict[str, Any],
    tool_results: dict[str, Any],
) -> float:
    """Score whether insights are relevant to available signals and non-trivial."""
    if not insights:
        return 0.0

    texts = [str(row.get("insight", "")).strip().lower() for row in insights if str(row.get("insight", "")).strip()]
    if not texts:
        return 0.0

    mentions_missing = any("missing" in t for t in texts)
    mentions_duplicate = any("duplicate" in t for t in texts)
    mentions_corr = any("correlation" in t or "r=" in t for t in texts)
    mentions_outlier = any("outlier" in t for t in texts)
    mentions_target = any("target" in t or "anova" in t or "class distribution" in t for t in texts)

    expected_checks: list[bool] = []
    expected_checks.append(bool(tool_results.get("missing_value_analysis")))
    expected_checks.append(bool(tool_results.get("duplicate_analysis")))
    expected_checks.append(bool(tool_results.get("correlation_analysis")))
    expected_checks.append(bool(tool_results.get("outlier_detection")))
    expected_checks.append(bool(tool_results.get("target_aware_analysis", {}).get("analysis_type") not in (None, "none")))

    covered_checks = [
        mentions_missing,
        mentions_duplicate,
        mentions_corr,
        mentions_outlier,
        mentions_target,
    ]
    expected_count = sum(1 for flag in expected_checks if flag)
    coverage_ratio = safe_div(
        sum(1 for i, expected in enumerate(expected_checks) if expected and covered_checks[i]),
        max(expected_count, 1),
        default=0.0,
    )

    id_like_cols = set(profile.get("id_like_columns", []))
    id_penalty_hits = 0
    for col in id_like_cols:
        needle = str(col).lower()
        if any(needle in t for t in texts):
            id_penalty_hits += 1
    id_penalty = min(0.35, 0.08 * id_penalty_hits)

    score = max(0.0, coverage_ratio - id_penalty) * 10.0
    return round(score, 3)


def _clarity_score(report_text: str, insights: list[dict[str, Any]]) -> float:
    if not report_text:
        return 0.0

    required_sections = [
        "## Dataset Name",
        "## Workflow Mode",
        "## Selected Tools",
        "## Dataset Overview",
        "## Data Quality Findings",
        "## Statistical Findings",
        "## Key Insights",
        "## Visualization Recommendations",
        "## Chart List",
        "## Limitations",
        "## Reproducibility Notes",
    ]

    section_ratio = safe_div(
        sum(1 for marker in required_sections if marker in report_text),
        len(required_sections),
        default=0.0,
    )

    readable_insights = sum(
        1
        for row in insights
        if 35 <= len(str(row.get("insight", "")).strip()) <= 260
    )
    insight_ratio = safe_div(readable_insights, max(len(insights), 1), default=0.0)

    source_diversity = safe_div(
        len({str(row.get("source_tool", "")).strip() for row in insights if row.get("source_tool")}),
        max(len(insights), 1),
        default=0.0,
    )

    return round((0.50 * section_ratio + 0.30 * insight_ratio + 0.20 * source_diversity) * 10.0, 3)


def _efficiency_score(
    runtime_seconds: float,
    number_of_tools_called: int,
    number_of_charts_generated: int,
    number_of_insights_generated: int,
    error_count: int,
) -> float:
    """Efficiency as useful outputs per runtime with complexity/error penalties."""
    if runtime_seconds <= 0:
        return 0.0

    useful_outputs = number_of_insights_generated + (0.5 * number_of_charts_generated)
    throughput = useful_outputs / max(runtime_seconds, 1e-6)
    log_throughput = math.log1p(max(throughput, 0.0))

    complexity_penalty = max(0, number_of_tools_called - 8) * 0.12
    error_penalty = error_count * 0.9

    score = (log_throughput * 4.8) - complexity_penalty - error_penalty
    return round(max(0.0, min(10.0, score)), 3)


def _validate_recommendation(
    profile: dict[str, Any],
    recommendation: dict[str, Any],
) -> tuple[bool, str | None]:
    numeric_cols = set(profile.get("numeric_columns", []))
    categorical_cols = set(profile.get("categorical_columns", []))
    row_count = int(profile.get("row_count", 0))
    chart_type = recommendation.get("chart_type")
    x_col = recommendation.get("x")
    y_col = recommendation.get("y")

    if chart_type == "missing_bar":
        has_missing = any(v > 0 for v in profile.get("missing_counts", {}).values())
        return has_missing, None if has_missing else "missing_bar_without_missing_values"

    if chart_type == "correlation_heatmap":
        valid_numeric = [c for c in numeric_cols if not _is_identifier_like(profile, c)]
        ok = len(valid_numeric) >= 2
        return ok, None if ok else "insufficient_non_identifier_numeric_columns_for_heatmap"

    if chart_type == "histogram":
        if x_col not in numeric_cols:
            return False, "histogram_non_numeric_column"
        if _is_identifier_like(profile, str(x_col)):
            return False, "histogram_identifier_like_column"
        unique_count = int(_column_stats(profile, str(x_col)).get("unique_count", 0))
        if unique_count < 3:
            return False, "histogram_low_unique_count"
        return True, None

    if chart_type == "scatter":
        if x_col not in numeric_cols or y_col not in numeric_cols:
            return False, "scatter_non_numeric_axis"
        if _is_identifier_like(profile, str(x_col)) or _is_identifier_like(profile, str(y_col)):
            return False, "scatter_identifier_like_axis"
        return True, None

    if chart_type == "bar":
        if x_col not in categorical_cols:
            return False, "bar_non_categorical_column"
        if _is_identifier_like(profile, str(x_col)):
            return False, "bar_identifier_like_column"
        stats = _column_stats(profile, str(x_col))
        unique_count = int(stats.get("unique_count", 0))
        unique_ratio = float(stats.get("unique_ratio", 0.0))
        singleton_ratio = float(stats.get("singleton_ratio", 0.0))
        max_unique_allowed = min(30, max(int(row_count * 0.2), 8))
        if unique_count > max_unique_allowed:
            return False, "bar_high_cardinality_column"
        if unique_ratio > 0.4:
            return False, "bar_high_unique_ratio_column"
        if singleton_ratio > 0.55 and unique_count > 12:
            return False, "bar_singleton_heavy_column"
        return True, None

    if chart_type == "target_bar_mean":
        if x_col not in categorical_cols:
            return False, "target_bar_mean_non_categorical_group"
        if _is_identifier_like(profile, str(x_col)):
            return False, "target_bar_mean_identifier_like_group"
        stats = _column_stats(profile, str(x_col))
        unique_count = int(stats.get("unique_count", 0))
        singleton_ratio = float(stats.get("singleton_ratio", 0.0))
        if unique_count > 12:
            return False, "target_bar_mean_high_cardinality_group"
        if singleton_ratio > 0.45 and unique_count > 8:
            return False, "target_bar_mean_singleton_heavy_group"
        return True, None

    if chart_type == "target_box":
        x_is_categorical = x_col in categorical_cols
        x_is_numeric_low_cardinality = (
            x_col in numeric_cols and int(_column_stats(profile, str(x_col)).get("unique_count", 0)) <= 12
        )
        if (not x_is_categorical and not x_is_numeric_low_cardinality) or y_col not in numeric_cols:
            return False, "target_box_invalid_axis_types"
        if _is_identifier_like(profile, str(y_col)):
            return False, "target_box_identifier_like_axis"
        if x_is_categorical and _is_identifier_like(profile, str(x_col)):
            return False, "target_box_identifier_like_axis"

        group_unique = int(_column_stats(profile, str(x_col)).get("unique_count", 0))
        if group_unique > 12:
            return False, "target_box_high_cardinality_group"
        return True, None

    return False, "unsupported_chart_type"


def _visualization_assessment(
    profile: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not recommendations:
        return {"score": 0.0, "invalid_recommendations": ["no_recommendations"], "valid_count": 0}

    valid_count = 0
    invalid_reasons: list[str] = []
    for rec in recommendations:
        is_valid, reason = _validate_recommendation(profile, rec)
        if is_valid:
            valid_count += 1
        elif reason:
            invalid_reasons.append(reason)

    base_score = safe_div(valid_count, len(recommendations), default=0.0) * 10.0
    return {
        "score": round(max(0.0, base_score), 3),
        "invalid_recommendations": invalid_reasons,
        "valid_count": valid_count,
    }


def evaluate_run(
    dataset_name: str,
    mode: str,
    strategy_mode: str | None,
    llm_label: str | None,
    selected_tools: list[str],
    tool_results: dict[str, Any],
    tool_log: list[dict[str, Any]],
    profile: dict[str, Any],
    runtime_seconds: float,
    report_text: str,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Compute evaluation metrics for a single run."""
    insight_payload = tool_results.get("insight_generation", {})
    insights = insight_payload.get("insights", [])

    chart_info = tool_results.get("chart_generation", {})
    chart_paths = chart_info.get("chart_paths", [])
    plot_failures = chart_info.get("plot_failures", [])
    recommendations = tool_results.get("visualization_recommendation", {}).get("recommendations", [])

    base_error_count = sum(1 for row in tool_log if row.get("status") == "error")
    error_count = int(base_error_count + len(plot_failures))

    viz_assessment = _visualization_assessment(profile, recommendations)

    quality_warnings = list(viz_assessment.get("invalid_recommendations", []))
    if fallback_used:
        quality_warnings.append("llm_fallback_used")
    if plot_failures:
        quality_warnings.append("chart_generation_failures_present")

    flesch = _flesch_reading_ease(report_text)

    result = {
        "dataset_name": dataset_name,
        "mode": mode,
        "strategy_mode": strategy_mode or mode,
        "llm_label": llm_label or "",
        "runtime_seconds": round(float(runtime_seconds), 4),
        "number_of_tools_called": len(selected_tools),
        "number_of_insights_generated": len(insights),
        "number_of_charts_generated": len(chart_paths),
        "completeness_score": _completeness_score(tool_results, profile),
        "insight_quality_score": _insight_quality_score(insights),
        "insight_relevance_score": _insight_relevance_score(insights, profile, tool_results),
        "redundancy_score": _redundancy_score(insights),
        "clarity_score": _clarity_score(report_text, insights),
        "flesch_reading_ease": round(flesch, 3),
        "readability_score": _flesch_to_ten_point(flesch),
        "visualization_suitability_score": float(viz_assessment["score"]),
        "error_count": error_count,
        "fallback_used": bool(fallback_used),
        "quality_warnings": quality_warnings,
    }
    result["efficiency_score"] = _efficiency_score(
        runtime_seconds=result["runtime_seconds"],
        number_of_tools_called=result["number_of_tools_called"],
        number_of_charts_generated=result["number_of_charts_generated"],
        number_of_insights_generated=result["number_of_insights_generated"],
        error_count=result["error_count"],
    )
    result["overall_quality_score"] = round(
        (
            0.20 * result["insight_quality_score"]
            + 0.16 * result["insight_relevance_score"]
            + 0.12 * result["redundancy_score"]
            + 0.14 * result["clarity_score"]
            + 0.08 * result["readability_score"]
            + 0.14 * result["visualization_suitability_score"]
            + 0.16 * result["efficiency_score"]
        ),
        3,
    )
    return result


def save_evaluation_results(evaluations: list[dict[str, Any]], output_root: Path) -> Path:
    """Save evaluation results into outputs/evaluation_results.csv."""
    ensure_dir(output_root)
    output_path = output_root / "evaluation_results.csv"
    save_records_csv(evaluations, output_path)
    return output_path


def _plot_metric(df: pd.DataFrame, group_col: str, metric: str, output_path: Path) -> None:
    grouped = df.groupby(group_col, as_index=False)[metric].mean(numeric_only=True)
    if grouped.empty:
        return
    grouped = grouped.sort_values(metric, ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar(grouped[group_col], grouped[metric])
    plt.title(f"Average {metric} by {group_col}")
    plt.xlabel(group_col)
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    samples = np.asarray(values, dtype=float)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(samples, size=len(samples), replace=True)
        boot_means[i] = float(np.mean(draw))
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - (alpha / 2)))
    return lower, upper


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    greater = 0
    lower = 0
    for x in a:
        greater += int(np.sum(x > b))
        lower += int(np.sum(x < b))
    return float((greater - lower) / max(len(a) * len(b), 1))


def _cliffs_delta_magnitude(delta: float) -> str:
    """Magnitude labels commonly used with Cliff's delta."""
    absolute = abs(float(delta))
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.33:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def _holm_bonferroni_adjusted(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (FWER control)."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted: list[float] = [1.0] * m
    running_max = 0.0
    for rank, (_, p_value) in enumerate(indexed):
        candidate = (m - rank) * float(p_value)
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = [1.0] * m
    for rank, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = adjusted_sorted[rank]
    return adjusted


def _kendalls_w(friedman_statistic: float, n_datasets: int, n_strategies: int) -> float:
    """Effect size for Friedman test: W = chi2 / (N * (k - 1))."""
    denominator = n_datasets * max(n_strategies - 1, 0)
    if denominator <= 0:
        return 0.0
    return float(friedman_statistic) / float(denominator)


def _kendalls_w_magnitude(kendalls_w: float) -> str:
    """Simple interpretation guidelines for Kendall's W."""
    absolute = abs(float(kendalls_w))
    if absolute < 0.1:
        return "negligible"
    if absolute < 0.3:
        return "small"
    if absolute < 0.5:
        return "moderate"
    return "large"


def _build_statistical_artifacts(
    df: pd.DataFrame,
    metric_cols: list[str],
    output_root: Path,
) -> None:
    tables_dir = ensure_dir(output_root / "comparison_tables")
    strategy_df = df.copy()
    if "strategy_mode" not in strategy_df.columns:
        strategy_df["strategy_mode"] = strategy_df["mode"]

    ci_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    friedman_rows: list[dict[str, Any]] = []

    for metric in metric_cols:
        if metric not in strategy_df.columns:
            continue
        metric_data = strategy_df[["dataset_name", "strategy_mode", metric]].dropna()
        if metric_data.empty:
            continue

        # Bootstrap CI by strategy
        for strategy, sub in metric_data.groupby("strategy_mode"):
            values = sub[metric].astype(float).values
            lower, upper = _bootstrap_mean_ci(values)
            ci_rows.append(
                {
                    "metric": metric,
                    "strategy_mode": strategy,
                    "sample_size": len(values),
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
                    "bootstrap_ci_95_low": round(lower, 4),
                    "bootstrap_ci_95_high": round(upper, 4),
                }
            )

        # Pairwise Wilcoxon signed-rank tests by dataset pairing
        pivot = metric_data.pivot_table(index="dataset_name", columns="strategy_mode", values=metric, aggfunc="mean")
        strategies = [str(col) for col in pivot.columns]
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                a_name = strategies[i]
                b_name = strategies[j]
                paired = pivot[[a_name, b_name]].dropna()
                if paired.empty:
                    continue
                a_vals = paired[a_name].astype(float).values
                b_vals = paired[b_name].astype(float).values
                n = len(a_vals)
                mean_a = float(np.mean(a_vals))
                mean_b = float(np.mean(b_vals))
                mean_diff = mean_a - mean_b
                if n >= 2:
                    if np.allclose(a_vals, b_vals):
                        p_value = 1.0
                    else:
                        try:
                            w_res = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
                            p_value = float(w_res.pvalue)
                        except Exception:
                            p_value = float("nan")
                else:
                    p_value = float("nan")
                delta = _cliffs_delta(a_vals, b_vals)
                pairwise_rows.append(
                    {
                        "metric": metric,
                        "strategy_a": a_name,
                        "strategy_b": b_name,
                        "n_datasets": n,
                        "mean_a": round(mean_a, 4),
                        "mean_b": round(mean_b, 4),
                        "mean_diff_a_minus_b": round(mean_diff, 4),
                        "wilcoxon_p_value": round(p_value, 6) if not math.isnan(p_value) else "",
                        "cliffs_delta": round(delta, 4),
                        "cliffs_delta_magnitude": _cliffs_delta_magnitude(delta),
                        "better_strategy": a_name if mean_a > mean_b else b_name if mean_b > mean_a else "tie",
                    }
                )

        # Friedman test across >=3 strategies
        if len(strategies) >= 3:
            complete = pivot.dropna()
            if len(complete) >= 2:
                rowwise_equal = np.allclose(
                    complete.values,
                    complete.values[:, [0]],
                )
                if rowwise_equal:
                    friedman_rows.append(
                        {
                            "metric": metric,
                            "n_datasets": int(len(complete)),
                            "n_strategies": int(len(strategies)),
                            "friedman_statistic": 0.0,
                            "friedman_p_value": 1.0,
                            "kendalls_w": 0.0,
                            "kendalls_w_magnitude": "negligible",
                        }
                    )
                    continue
                try:
                    vectors = [complete[s].astype(float).values for s in strategies]
                    f_res = stats.friedmanchisquare(*vectors)
                    kendalls_w = _kendalls_w(
                        friedman_statistic=float(f_res.statistic),
                        n_datasets=int(len(complete)),
                        n_strategies=int(len(strategies)),
                    )
                    friedman_rows.append(
                        {
                            "metric": metric,
                            "n_datasets": int(len(complete)),
                            "n_strategies": int(len(strategies)),
                            "friedman_statistic": round(float(f_res.statistic), 6),
                            "friedman_p_value": round(float(f_res.pvalue), 6),
                            "kendalls_w": round(kendalls_w, 6),
                            "kendalls_w_magnitude": _kendalls_w_magnitude(kendalls_w),
                        }
                    )
                except Exception:
                    pass

    if pairwise_rows:
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df["wilcoxon_p_holm"] = np.nan
        pairwise_df["reject_h0_alpha_0_05"] = False

        for metric, group in pairwise_df.groupby("metric", sort=False):
            idxs = list(group.index)
            valid_indices: list[int] = []
            valid_pvals: list[float] = []
            for idx in idxs:
                raw = pairwise_df.at[idx, "wilcoxon_p_value"]
                if raw in ("", None):
                    continue
                try:
                    p_val = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isnan(p_val):
                    continue
                valid_indices.append(idx)
                valid_pvals.append(p_val)

            if not valid_pvals:
                continue

            adjusted = _holm_bonferroni_adjusted(valid_pvals)
            for local_i, idx in enumerate(valid_indices):
                adjusted_p = float(adjusted[local_i])
                pairwise_df.at[idx, "wilcoxon_p_holm"] = round(adjusted_p, 6)
                pairwise_df.at[idx, "reject_h0_alpha_0_05"] = bool(adjusted_p < 0.05)

        pairwise_rows = pairwise_df.to_dict(orient="records")

    save_records_csv(ci_rows, tables_dir / "strategy_metric_cis.csv")
    save_records_csv(pairwise_rows, tables_dir / "pairwise_significance.csv")
    save_records_csv(friedman_rows, tables_dir / "friedman_tests.csv")

    lines = [
        "# Statistical Significance Summary",
        "",
        "Methods: bootstrap 95% CI for strategy means, Wilcoxon signed-rank pairwise tests, and Friedman tests across strategies.",
        "",
    ]
    if friedman_rows:
        lines.append("## Friedman Tests")
        lines.append(
            tabulate(
                friedman_rows,
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )
        lines.append("")
    if pairwise_rows:
        lines.append("## Pairwise Wilcoxon (All Metrics)")
        lines.append(
            tabulate(
                pairwise_rows[:30],
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )
        lines.append("")
        lines.append(f"Full table: comparison_tables/pairwise_significance.csv ({len(pairwise_rows)} rows)")
        lines.append("")
    write_text(output_root / "statistical_significance_summary.md", "\n".join(lines))


def generate_comparison_artifacts(evaluations: list[dict[str, Any]], output_root: Path) -> None:
    """Create comparison tables, plots, summary markdown, and statistical artifacts."""
    if not evaluations:
        return

    ensure_dir(output_root)
    tables_dir = ensure_dir(output_root / "comparison_tables")
    plots_dir = ensure_dir(output_root / "comparison_plots")

    df = pd.DataFrame(evaluations)
    if "strategy_mode" not in df.columns:
        df["strategy_mode"] = df["mode"]
    df["strategy_mode"] = (
        df["strategy_mode"]
        .replace(["", "nan", "None", "none"], np.nan)
        .fillna(df["mode"])
    )
    if "llm_label" not in df.columns:
        df["llm_label"] = ""
    df["llm_label"] = df["llm_label"].replace(["nan", "None", "none"], "").fillna("")

    metric_cols = [
        "runtime_seconds",
        "number_of_tools_called",
        "number_of_insights_generated",
        "number_of_charts_generated",
        "completeness_score",
        "insight_quality_score",
        "insight_relevance_score",
        "redundancy_score",
        "clarity_score",
        "flesch_reading_ease",
        "readability_score",
        "visualization_suitability_score",
        "efficiency_score",
        "overall_quality_score",
        "error_count",
    ]

    # Mode-level tables (captures each LLM variant separately)
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        pivot_mode = df.pivot_table(index="dataset_name", columns="mode", values=metric, aggfunc="mean")
        pivot_mode.reset_index().to_csv(tables_dir / f"{metric}_by_dataset_mode.csv", index=False)
        _plot_metric(df, "mode", metric, plots_dir / f"avg_{metric}_by_mode.png")

    # Strategy-level tables (fixed vs rule vs llm)
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        pivot_strategy = df.pivot_table(index="dataset_name", columns="strategy_mode", values=metric, aggfunc="mean")
        pivot_strategy.reset_index().to_csv(tables_dir / f"{metric}_by_dataset_strategy.csv", index=False)
        _plot_metric(df, "strategy_mode", metric, plots_dir / f"avg_{metric}_by_strategy.png")

    _build_statistical_artifacts(df, metric_cols, output_root)

    # Strategy summary
    sortable = df.sort_values(["dataset_name", "strategy_mode", "mode"]).copy()
    summary_table = tabulate(sortable.to_dict(orient="records"), headers="keys", tablefmt="github", showindex=False)
    strategy_means = df.groupby("strategy_mode", as_index=False).mean(numeric_only=True)

    lower_is_better = {"runtime_seconds", "error_count", "number_of_tools_called"}
    best_lines = []
    for metric in metric_cols:
        if metric not in strategy_means.columns:
            continue
        series = strategy_means[["strategy_mode", metric]].dropna()
        if series.empty:
            continue
        if metric in lower_is_better:
            best_row = series.loc[series[metric].idxmin()]
        else:
            best_row = series.loc[series[metric].idxmax()]
        best_lines.append(f"- {metric}: {best_row['strategy_mode']} ({best_row[metric]:.4f})")

    overall_rank = (
        strategy_means[["strategy_mode", "overall_quality_score"]]
        .sort_values("overall_quality_score", ascending=False)
        .reset_index(drop=True)
    )
    rank_lines = [
        f"{idx + 1}. {row['strategy_mode']} ({row['overall_quality_score']:.3f})"
        for idx, row in overall_rank.iterrows()
    ]

    # Optional: llm-only leaderboard when multiple llm variants exist
    llm_rows = df[df["strategy_mode"] == "llm"].copy()
    llm_lines: list[str] = []
    if not llm_rows.empty and llm_rows["llm_label"].astype(str).str.len().gt(0).any():
        llm_means = (
            llm_rows.groupby("llm_label", as_index=False)
            .mean(numeric_only=True)
            .sort_values("overall_quality_score", ascending=False)
        )
        llm_lines = [
            "## LLM Variant Ranking (By Mean overall_quality_score)",
            tabulate(llm_means.to_dict(orient="records"), headers="keys", tablefmt="github", showindex=False),
            "",
        ]

    interpretation = [
        "- Fixed pipeline provides a deterministic baseline.",
        "- Rule-based mode provides adaptive yet non-LLM tool selection.",
        "- LLM mode quality should be interpreted with efficiency and fallback behavior together.",
        "- Statistical artifacts include bootstrap CIs, Wilcoxon pairwise tests, and Friedman omnibus tests.",
    ]

    fallback_lines: list[str] = []
    if "fallback_used" in df.columns:
        fallback_summary = (
            df.groupby("strategy_mode", as_index=False)["fallback_used"]
            .mean(numeric_only=True)
            .rename(columns={"fallback_used": "fallback_rate"})
        )
        fallback_summary["fallback_rate"] = fallback_summary["fallback_rate"].astype(float).round(4)
        fallback_lines = [
            "## LLM Fallback Rate (Lower Is Better)",
            tabulate(
                fallback_summary.to_dict(orient="records"),
                headers="keys",
                tablefmt="github",
                showindex=False,
            ),
            "",
        ]

    summary_lines = [
        "# Comparison Summary",
        "",
        "## Comparison Table Across Runs",
        summary_table,
        "",
        "## Best Strategy per Metric",
        *best_lines,
        "",
        "## Interpretation of Findings",
        *interpretation,
        "",
        "## Overall Strategy Ranking (By Mean overall_quality_score)",
        *rank_lines,
        "",
        *fallback_lines,
        *llm_lines,
        "## Report Discussion",
        "This benchmark compares lightweight agentic EDA workflows across consistent tabular datasets. ",
        "Fixed and rule-based approaches ensure reproducible execution without external model dependence, ",
        "while LLM-driven planning can improve adaptivity but introduces response-validity and efficiency trade-offs.",
        "",
    ]

    write_text(output_root / "comparison_summary.md", "\n".join(summary_lines))
