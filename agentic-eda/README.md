# Evaluating Lightweight Agentic Workflows for Automated Exploratory Data Analysis on Tabular Datasets

This project provides a reproducible experimental framework for comparing lightweight automated EDA workflows on tabular CSV datasets.  
It evaluates whether adaptive orchestration (rule-based or LLM-guided) provides measurable value over a deterministic fixed pipeline, while keeping execution safe, auditable, and suitable for academic reporting.

## Why This Project
- Establishes a consistent benchmark for automated EDA workflow strategies.
- Produces report-ready artifacts (tables, plots, logs, and markdown reports).
- Supports technical analysis of quality, efficiency, and robustness trade-offs.

## Key Features
- `Fixed Pipeline Baseline`: always runs the same safe tool sequence.
- `Rule-Based Agent`: chooses tools deterministically from dataset profile properties.
- `LLM-Based Agent`: plans tools via strict JSON output, validated against an allowlisted tool registry.
- Safe fallback from LLM mode to rule-based mode when planning is invalid/unavailable.
- Unified evaluation framework with per-run and cross-run comparison artifacts.
- Statistical comparison outputs (bootstrap confidence intervals, Wilcoxon tests, Holm correction, Friedman + Kendall's W).

## Project Structure
```text
agentic-eda/
├─ agents/                      # Orchestration strategies (fixed, rule-based, llm)
├─ data/
│  └─ sample/                   # Benchmark datasets used by experiments
├─ outputs/                     # Generated run artifacts and comparison artifacts
├─ scripts/                     # Utility scripts (dataset preparation, multi-model runner)
├─ src/                         # Core EDA tools, profiling, reporting, evaluation modules
├─ tests/                       # Test cases
├─ config.py                    # Runtime configuration (env-driven)
├─ main.py                      # Single-dataset execution entrypoint
├─ run_all.py                   # Multi-dataset benchmark execution
└─ requirements.txt             # Python dependencies
```

## Installation
### 1) Create and activate a Python environment
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Copy `.env.example` to `.env`, then set values as needed (LLM provider/model/API settings, timeout, seed).

## Datasets Used
The benchmark uses these tabular datasets:
- `titanic_style`
- `diamonds`
- `adult_income`

Dataset preparation can be run with:
```bash
python scripts/download_sample_datasets.py
```

## How to Run Experiments
### Run all modes across all sample datasets
```bash
python run_all.py
```

### Run one mode on one dataset
```bash
python main.py --dataset data/sample/titanic_style.csv --mode fixed
python main.py --dataset data/sample/titanic_style.csv --mode rule
python main.py --dataset data/sample/titanic_style.csv --mode llm --llm-provider ollama --llm-model qwen2.5:3b
```

### Run multi-model Ollama benchmark
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_ollama_models.ps1
```

Optional fresh aggregate comparison:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_ollama_models.ps1 -FreshComparison
```

## Outputs and Artifacts
Each run generates structured artifacts under `outputs/{dataset}/{mode_or_model}/`, including:
- `profile.json` (dataset profile)
- `tool_results.json` (tool outputs)
- `tool_log.csv` (step-level execution log with runtime/status/errors)
- `insights.csv` (generated insights)
- `visualization_recommendations.csv` (recommended chart plans)
- `report.md` (narrative EDA report)
- `evaluation.json` (run-level evaluation metrics)
- `charts/*.png` (generated visualizations)

Global comparison artifacts include:
- `outputs/evaluation_results.csv`
- `outputs/comparison_summary.md`
- `outputs/statistical_significance_summary.md`
- `outputs/comparison_tables/*.csv`
- `outputs/comparison_plots/*.png`

## Reproducibility
- Deterministic components are used where applicable (fixed/rule strategies, controlled seed in config and data generation path).
- LLM planning is constrained by strict JSON schema expectations and safe tool allowlisting.
- Invalid LLM plans automatically trigger rule-based fallback, preserving run completion and comparability.
- All major outputs are written to disk in machine-readable formats for independent verification.

## Academic Use
This repository is structured to support MSc-level methodology, experimental setup, results reporting, and replication for tabular automated EDA workflow research.
