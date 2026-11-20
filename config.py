"""Tiny settings loader. Reads a local .env file (never committed) and the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from ``path`` into os.environ (existing variables win)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    data_path: str
    classifier_dir: str
    ngram_order: int


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_path=os.environ.get("DATA_PATH", "data/app.json"),
        classifier_dir=os.environ.get("CLASSIFIER_DIR", "models/clause-gpt2"),
        ngram_order=int(os.environ.get("NGRAM_ORDER", "3")),
    )
