from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OmniBrief with temporary LLM model overrides. "
            "Reports and traces are saved using the normal OmniBrief artifact naming."
        )
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help="Model used by browser-use extraction agents, for example 'gpt-4o-mini'.",
    )
    parser.add_argument(
        "--summary-model",
        required=True,
        help="Model used for the final executive summary, for example 'gpt-4o-mini'.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override for this test run.",
    )
    return parser.parse_args()


def configure_test_environment(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    os.environ["LLM_MODEL"] = args.llm_model
    os.environ["SUMMARY_MODEL"] = args.summary_model

    if args.base_url:
        os.environ["BASE_URL"] = args.base_url

    os.environ.setdefault("AGENT_USE_VISION", "false")
    os.environ.setdefault("AGENT_USE_JUDGE", "false")


def import_omnibrief_main():
    sys.path.insert(0, str(SRC_DIR))

    # Import after environment overrides so config.py and formatter.py pick up the test models.
    from main import main

    return main


def print_test_banner(args: argparse.Namespace) -> None:
    print("Running OmniBrief model test")
    print(f"Extraction model: {args.llm_model}")
    print(f"Summary model: {args.summary_model}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Artifacts will be saved under output/ and traces/ using the standard run ID.")


if __name__ == "__main__":
    cli_args = parse_args()
    configure_test_environment(cli_args)
    print_test_banner(cli_args)
    asyncio.run(import_omnibrief_main()())
