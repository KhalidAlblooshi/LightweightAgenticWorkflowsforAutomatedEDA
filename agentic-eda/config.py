"""Configuration for the Agentic EDA project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings loaded from environment variables."""

    project_root: Path
    data_dir: Path
    sample_data_dir: Path
    outputs_dir: Path

    random_seed: int

    llm_provider: str
    llm_timeout_seconds: int

    ollama_base_url: str
    ollama_model: str

    openai_base_url: str
    openai_api_key: str
    openai_model: str



def get_settings() -> Settings:
    """Build settings object from environment variables with safe defaults."""
    project_root = Path(__file__).resolve().parent

    return Settings(
        project_root=project_root,
        data_dir=project_root / "data",
        sample_data_dir=project_root / "data" / "sample",
        outputs_dir=project_root / "outputs",
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


SETTINGS = get_settings()
