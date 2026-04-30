"""Configuration module — loads settings from .env or environment variables."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    DATA_DIR: Path = Path("data/sample")
    OUTPUT_DIR: Path = Path("outputs")


config = Config()
