from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

try:
    from .browser_agent import extract_data_with_trace
    from .config import Target, load_targets
    from .formatter import generate_markdown_report
    from .telemetry import TokenUsage
except ImportError:
    from browser_agent import extract_data_with_trace
    from config import Target, load_targets
    from formatter import generate_markdown_report
    from telemetry import TokenUsage


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _target_to_dict(target: Target) -> dict[str, str]:
    return asdict(target)


async def main() -> None:
    configure_logging()
    run_start = time.perf_counter()
    total_usage = TokenUsage()

    logger.info("Loading briefing targets")
    targets = load_targets()

    if not targets:
        logger.warning("No targets found in targets.json. Nothing to process.")
        return

    results: list[dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        target_dict = _target_to_dict(target)
        logger.info("[%s/%s] Processing %s: %s", index, len(targets), target.name, target.url)

        extraction = await extract_data_with_trace(target_dict)
        total_usage.add(extraction.token_usage)

        results.append(
            {
                "name": target.name,
                "url": target.url,
                "status": extraction.status,
                "content": extraction.content,
                "trace_path": str(extraction.trace_path) if extraction.trace_path else "",
                "prompt_tokens": extraction.token_usage.prompt_tokens,
                "completion_tokens": extraction.token_usage.completion_tokens,
                "total_tokens": extraction.token_usage.total_tokens,
                "execution_time_seconds": round(extraction.execution_time_seconds, 3),
            }
        )

        logger.info(
            "[%s/%s] Finished %s with status=%s tokens=%s trace=%s",
            index,
            len(targets),
            target.name,
            extraction.status,
            extraction.token_usage.total_tokens,
            extraction.trace_path,
        )

    total_execution_time = time.perf_counter() - run_start
    report_path = generate_markdown_report(
        results,
        telemetry={
            "prompt_tokens": total_usage.prompt_tokens,
            "completion_tokens": total_usage.completion_tokens,
            "total_tokens": total_usage.total_tokens,
            "execution_time_seconds": round(total_execution_time, 3),
        },
    )
    logger.info("Markdown briefing generated: %s", report_path)
    logger.info(
        "Run telemetry: prompt_tokens=%s completion_tokens=%s total_tokens=%s execution_time_seconds=%.3f",
        total_usage.prompt_tokens,
        total_usage.completion_tokens,
        total_usage.total_tokens,
        total_execution_time,
    )


if __name__ == "__main__":
    asyncio.run(main())
