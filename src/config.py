from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
TARGETS_FILE = BASE_DIR / "targets.json"


load_dotenv(ENV_FILE)


@dataclass
class Target:
    name: str
    url: str
    extraction_prompt: str


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or not value.strip():
        return default
    return float(value.strip())


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _parse_csv(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None or not value.strip():
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


API_KEY = os.getenv("API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
LLM_TEMPERATURE = _parse_float(os.getenv("LLM_TEMPERATURE"))
LLM_MAX_TOKENS = _parse_int(os.getenv("LLM_MAX_TOKENS"))
LLM_TIMEOUT = _parse_int(os.getenv("LLM_TIMEOUT"), default=60)
HEADLESS_MODE = _parse_bool(os.getenv("HEADLESS_MODE"), default=True)
AGENT_USE_VISION = _parse_bool(os.getenv("AGENT_USE_VISION"), default=False)
AGENT_USE_JUDGE = _parse_bool(os.getenv("AGENT_USE_JUDGE"), default=False)
AGENT_MAX_FAILURES = _parse_int(os.getenv("AGENT_MAX_FAILURES"), default=3)
AGENT_CONTEXT_MINIMIZATION = _parse_bool(os.getenv("AGENT_CONTEXT_MINIMIZATION"), default=True)
AGENT_INCLUDE_ATTRIBUTES = _parse_csv(
    os.getenv("AGENT_INCLUDE_ATTRIBUTES"),
    default=["href", "src", "id", "aria-label", "title", "alt"],
)


def load_targets(file_path: str | Path = TARGETS_FILE) -> List[Target]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_targets = json.load(file)

    if not isinstance(raw_targets, list):
        raise ValueError("targets.json must contain a JSON array.")

    targets: List[Target] = []
    for index, item in enumerate(raw_targets):
        if not isinstance(item, dict):
            raise ValueError(f"Target at index {index} must be a JSON object.")

        missing_keys = {"name", "url", "extraction_prompt"} - item.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Target at index {index} is missing keys: {missing}")

        targets.append(
            Target(
                name=str(item["name"]).strip(),
                url=str(item["url"]).strip(),
                extraction_prompt=str(item["extraction_prompt"]).strip(),
            )
        )
    return targets
