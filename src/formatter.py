from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from .config import API_KEY, BASE_URL, LLM_TIMEOUT
except ImportError:
    from config import API_KEY, BASE_URL, LLM_TIMEOUT


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip()
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.2"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "600"))
SUMMARY_MAX_INPUT_CHARS = int(os.getenv("SUMMARY_MAX_INPUT_CHARS", "12000"))

logger = logging.getLogger(__name__)
ARTIFACT_PREFIX = "omnibrief-morning-briefing"


def _format_artifact_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def _build_run_id(timestamp: datetime) -> str:
    return f"{ARTIFACT_PREFIX}_{_format_artifact_timestamp(timestamp)}"


def _format_local_link(path_value: str | Path, output_file: Path, label: str | None = None) -> str:
    path = Path(path_value)
    link_label = label or path.name
    try:
        link_target = os.path.relpath(path, start=output_file.parent)
    except ValueError:
        link_target = str(path)
    return f"[{link_label}]({link_target})"


def _build_report_header(
    *,
    timestamp: datetime,
    run_id: str,
    trace_dir: str | Path | None,
    output_file: Path,
) -> str:
    lines = [
        "# Morning Briefing",
        "",
        f"Generated on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run ID: `{run_id}`",
    ]

    if trace_dir:
        lines.append(f"Trace folder: {_format_local_link(trace_dir, output_file)}")

    return "\n".join(lines) + "\n"


def _format_result_section(result: dict[str, Any], output_file: Path) -> str:
    target_index = result.get("target_index")
    name = str(result.get("name", "Unnamed Target")).strip() or "Unnamed Target"
    url = str(result.get("url", "")).strip()
    status = str(result.get("status", "unknown")).strip().lower()
    content = str(result.get("content", "")).strip()
    trace_path = str(result.get("trace_path", "")).strip()

    if status == "success":
        body = content or "No content was returned."
    else:
        body = f"Extraction failed.\n\n{content or 'No additional error details were provided.'}"

    heading_prefix = f"{target_index}. " if target_index else ""
    link_line = f"Source: [{url}]({url})" if url else "Source: URL not provided"
    metadata_lines = [
        link_line,
        f"Status: `{status}`",
    ]

    if trace_path:
        metadata_lines.append(f"Trace file: {_format_local_link(trace_path, output_file)}")

    metadata = "\n\n".join(metadata_lines)
    return f"## {heading_prefix}{name}\n\n{metadata}\n\n{body}\n"


def _build_summary_llm() -> ChatOpenAI:
    llm_kwargs: dict[str, Any] = {
        "model": SUMMARY_MODEL,
        "api_key": API_KEY,
        "temperature": SUMMARY_TEMPERATURE,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }

    if BASE_URL:
        llm_kwargs["base_url"] = BASE_URL
    if LLM_TIMEOUT is not None:
        llm_kwargs["timeout"] = LLM_TIMEOUT

    return ChatOpenAI(**llm_kwargs)


def _successful_result_text(raw_results: list[dict[str, Any]]) -> str:
    successful_sections: list[str] = []

    for result in raw_results:
        if str(result.get("status", "")).lower() != "success":
            continue

        name = str(result.get("name", "Unnamed Target")).strip() or "Unnamed Target"
        content = str(result.get("content", "")).strip()
        if content:
            successful_sections.append(f"Source: {name}\n{content}")

    combined = "\n\n---\n\n".join(successful_sections)
    return combined[:SUMMARY_MAX_INPUT_CHARS]


async def generate_executive_summary(raw_results: list[dict[str, Any]]) -> str:
    summary_input = _successful_result_text(raw_results)

    if not summary_input:
        return "No successful extractions were available to summarize."

    if not API_KEY:
        logger.warning("Skipping executive summary because API_KEY is not set.")
        return "Executive summary unavailable because API_KEY is not set."

    prompt = (
        "Write a cohesive, 3-paragraph Morning Briefing Executive Summary from the extracted source material below. "
        "Synthesize the most important developments across sources, avoid bullet points, avoid mentioning implementation details, "
        "and do not invent facts that are not present in the source material.\n\n"
        f"{summary_input}"
    )

    try:
        llm = _build_summary_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are an executive briefing analyst. Write concise, factual morning briefings "
                        "for a busy reader who needs the big picture quickly."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        return str(response.content).strip()
    except Exception as exc:
        logger.exception("Executive summary generation failed.")
        return f"Executive summary unavailable due to summarization error: {exc}"


def _format_executive_summary_section(summary: str) -> str:
    return f"## Morning Briefing Executive Summary\n\n{summary.strip()}\n"


def _format_telemetry_section(telemetry: dict[str, Any] | None) -> str:
    telemetry = telemetry or {}
    prompt_tokens = int(telemetry.get("prompt_tokens", 0) or 0)
    completion_tokens = int(telemetry.get("completion_tokens", 0) or 0)
    total_tokens = int(telemetry.get("total_tokens", 0) or 0)
    execution_time_seconds = float(telemetry.get("execution_time_seconds", 0.0) or 0.0)

    return (
        "## Telemetry & Costs\n\n"
        f"- Total prompt tokens: {prompt_tokens}\n"
        f"- Total completion tokens: {completion_tokens}\n"
        f"- Total combined tokens: {total_tokens}\n"
        f"- Total execution time: {execution_time_seconds:.3f} seconds\n"
    )


async def generate_markdown_report(
    results: list[dict[str, Any]],
    telemetry: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
    generated_at: datetime | None = None,
    trace_dir: str | Path | None = None,
) -> Path:
    timestamp = generated_at or datetime.now()
    report_run_id = run_id or _build_run_id(timestamp)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{report_run_id}.md"

    executive_summary = await generate_executive_summary(results)

    sections = [
        _build_report_header(
            timestamp=timestamp,
            run_id=report_run_id,
            trace_dir=trace_dir,
            output_file=output_file,
        ),
        _format_executive_summary_section(executive_summary),
    ]
    for result in results:
        sections.append(_format_result_section(result, output_file))
    sections.append(_format_telemetry_section(telemetry))

    markdown = "\n".join(sections).strip() + "\n"
    output_file.write_text(markdown, encoding="utf-8")

    return output_file
