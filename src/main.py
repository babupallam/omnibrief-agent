from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .browser_agent import extract_data_with_trace
    from .config import BASE_DIR, Target, load_targets
    from .formatter import generate_markdown_report
    from .telemetry import TokenUsage
except ImportError:
    from browser_agent import extract_data_with_trace
    from config import BASE_DIR, Target, load_targets
    from formatter import generate_markdown_report
    from telemetry import TokenUsage


logger = logging.getLogger(__name__)
ARTIFACT_PREFIX = "omnibrief-morning-briefing"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _target_to_dict(target: Target) -> dict[str, str]:
    return asdict(target)


def _build_run_id(run_started_at: datetime) -> str:
    return f"{ARTIFACT_PREFIX}_{run_started_at.strftime('%Y-%m-%d_%H-%M-%S')}"


def _create_run_trace_dir(run_id: str) -> Path:
    trace_dir = BASE_DIR / "traces" / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


async def _process_target(index: int, total: int, target: Target, trace_dir: Path) -> dict[str, Any]:
    target_dict = _target_to_dict(target)
    target_start = time.perf_counter()

    logger.info("[%s/%s] Processing %s: %s", index, total, target.name, target.url)

    try:
        extraction = await extract_data_with_trace(target_dict, trace_dir=trace_dir)
        result = {
            "target_index": index,
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

        logger.info(
            "[%s/%s] Finished %s with status=%s tokens=%s trace=%s",
            index,
            total,
            target.name,
            extraction.status,
            extraction.token_usage.total_tokens,
            extraction.trace_path,
        )
        return result
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - target_start
        logger.exception("[%s/%s] Timed out while processing %s", index, total, target.name)
        return {
            "target_index": index,
            "name": target.name,
            "url": target.url,
            "status": "failure",
            "content": f"Extraction timed out for {target.name}.",
            "trace_path": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "execution_time_seconds": round(elapsed, 3),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - target_start
        logger.exception("[%s/%s] Failed while processing %s", index, total, target.name)
        return {
            "target_index": index,
            "name": target.name,
            "url": target.url,
            "status": "failure",
            "content": f"Extraction failed for {target.name}: {exc}",
            "trace_path": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "execution_time_seconds": round(elapsed, 3),
        }


async def main() -> None:
    configure_logging()
    run_started_at = datetime.now()
    run_start = time.perf_counter()
    total_usage = TokenUsage()

    logger.info("Loading briefing targets")
    targets = load_targets()

    if not targets:
        logger.warning("No targets found in targets.json. Nothing to process.")
        return

    run_id = _build_run_id(run_started_at)
    trace_dir = _create_run_trace_dir(run_id)
    logger.info("Run ID: %s", run_id)
    logger.info("Saving trace files for this run under %s", trace_dir)
    logger.info("Processing %s targets concurrently", len(targets))
    tasks = [
        _process_target(index, len(targets), target, trace_dir)
        for index, target in enumerate(targets, start=1)
    ]
    results = await asyncio.gather(*tasks)

    for result in results:
        total_usage.add(
            TokenUsage(
                prompt_tokens=int(result.get("prompt_tokens", 0) or 0),
                completion_tokens=int(result.get("completion_tokens", 0) or 0),
                total_tokens=int(result.get("total_tokens", 0) or 0),
            )
        )

    total_execution_time = time.perf_counter() - run_start
    report_path = await generate_markdown_report(
        results,
        run_id=run_id,
        generated_at=run_started_at,
        trace_dir=trace_dir,
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
