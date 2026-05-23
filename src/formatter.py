from __future__ import annotations

import logging
import os
import re
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
SUMMARY_HEADING = "Morning Briefing Executive Summary"


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


def _clean_extracted_content(content: str) -> str:
    lines = [line.rstrip() for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _extract_labeled_value(lines: list[str], labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(rf"^\s*(?:[-*]\s*)?(?:\*\*)?(?:{label_pattern})(?:\*\*)?\s*[:\-–]\s*(.+)\s*$", re.I)

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def _is_temperature_line(line: str) -> bool:
    return bool(re.search(r"(?:°|(?:\b-?\d+\s*(?:c|f)\b))", line, re.I))


def _is_percentage_line(line: str) -> bool:
    return "%" in line or bool(re.search(r"\b(?:rain|precipitation|showers?)\b", line, re.I))


def _format_metric(label: str, value: str | None) -> str:
    return f"**{label}:** {value if value else 'not available'}"


def _format_weather_content(content: str) -> str:
    cleaned = _clean_extracted_content(content)
    lines = [line.strip(" -•\t") for line in cleaned.split("\n") if line.strip()]
    weather_label_pattern = re.compile(
        r"^\s*(?:\*\*)?"
        r"(?:high|high temperature|today's high|maximum|max|low|low temperature|today's low|minimum|min|"
        r"current conditions|conditions|condition|weather|precipitation|chance of precipitation|chance of rain|rain|"
        r"wind|winds|wind speed|alert|alerts|warning|warnings)"
        r"(?:\*\*)?\s*[:\-–]",
        re.I,
    )

    high = _extract_labeled_value(lines, ("high", "high temperature", "today's high", "maximum", "max"))
    low = _extract_labeled_value(lines, ("low", "low temperature", "today's low", "minimum", "min"))
    conditions = _extract_labeled_value(lines, ("current conditions", "conditions", "condition", "weather"))
    precipitation = _extract_labeled_value(
        lines,
        ("precipitation", "chance of precipitation", "chance of rain", "rain"),
    )
    wind = _extract_labeled_value(lines, ("wind", "winds", "wind speed"))
    alerts = _extract_labeled_value(lines, ("alert", "alerts", "warning", "warnings"))

    temperature_lines = [line for line in lines if _is_temperature_line(line)]
    if not high and temperature_lines:
        high = temperature_lines[0]
    if not low and len(temperature_lines) > 1:
        low = temperature_lines[1]

    if not precipitation:
        precipitation = next((line for line in lines if _is_percentage_line(line) and not _is_temperature_line(line)), None)

    if not conditions:
        conditions = next(
            (
                line
                for line in lines
                if not _is_temperature_line(line)
                and not _is_percentage_line(line)
                and not re.search(r"^\w+(?:\s+\w+){0,3}\s*[:\-–]", line)
            ),
            None,
        )

    known_values = {value for value in (high, low, conditions, precipitation, wind, alerts) if value}
    additional_details = [
        line
        for line in lines
        if line not in known_values and not weather_label_pattern.match(line)
    ]

    formatted_lines = [
        _format_metric("High", high),
        _format_metric("Low", low),
        _format_metric("Current Conditions", conditions),
        _format_metric("Precipitation", precipitation),
    ]

    if wind:
        formatted_lines.append(_format_metric("Wind", wind))
    if alerts:
        formatted_lines.append(_format_metric("Alerts", alerts))
    if additional_details:
        formatted_lines.append(f"**Additional Details:** {'; '.join(additional_details)}")

    return "\n".join(formatted_lines)


def _format_hacker_news_content(content: str) -> str:
    cleaned = _clean_extracted_content(content)
    formatted_lines: list[str] = []
    hn_pattern = re.compile(
        r"^\s*(?P<index>\d+)[.)]\s*(?:\*\*)?(?P<title>.+?)(?:\*\*)?\s*[–-]\s*"
        r"(?P<points>\d+)\s+points?,\s*(?P<comments>\d+)\s+comments?\s*[–-]\s*"
        r"(?:Community reaction:\s*)?(?P<reaction>.+)\s*$",
        re.I,
    )

    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        match = hn_pattern.match(stripped)
        if not match:
            formatted_lines.append(stripped)
            continue

        title = match.group("title").strip().strip("*")
        formatted_lines.extend(
            [
                f"{match.group('index')}. **{title}**",
                f"   **Points:** {match.group('points')}",
                f"   **Comments:** {match.group('comments')}",
                f"   **Community Reaction:** {match.group('reaction').strip()}",
            ]
        )

    return "\n".join(formatted_lines) if formatted_lines else cleaned


def _format_success_body(name: str, url: str, content: str) -> str:
    cleaned = _clean_extracted_content(content)
    lowered_name = name.lower()
    lowered_url = url.lower()

    if "weather" in lowered_name or "weather" in lowered_url:
        return _format_weather_content(cleaned)
    if "hacker news" in lowered_name or "news.ycombinator.com" in lowered_url:
        return _format_hacker_news_content(cleaned)
    return cleaned or "No content was returned."


def _format_unavailable_block(content: str) -> str:
    reason = _clean_extracted_content(content) or "No additional error details were provided."
    return f"Status: Not Available\n\nReason: {reason}"


def _strip_duplicate_summary_heading(summary: str) -> str:
    cleaned = _clean_extracted_content(summary)
    heading_pattern = re.compile(
        rf"^\s*(?:#+\s*)?(?:\*\*)?{re.escape(SUMMARY_HEADING)}(?:\*\*)?\s*:?\s*",
        re.I,
    )

    while True:
        updated = heading_pattern.sub("", cleaned, count=1).lstrip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _format_result_section(result: dict[str, Any], output_file: Path) -> str:
    target_index = result.get("target_index")
    name = str(result.get("name", "Unnamed Target")).strip() or "Unnamed Target"
    url = str(result.get("url", "")).strip()
    status = str(result.get("status", "unknown")).strip().lower()
    content = _clean_extracted_content(str(result.get("content", "")).strip())
    trace_path = str(result.get("trace_path", "")).strip()

    if status == "success":
        status_line = "**Status:** Available"
        body = _format_success_body(name, url, content)
    else:
        status_line = "Status: Not Available"
        body = _format_unavailable_block(content)

    heading_prefix = f"{target_index}. " if target_index else ""
    link_line = f"**Source:** [{url}]({url})" if url else "**Source:** URL not provided"
    metadata_lines = [
        link_line,
    ]
    if status == "success":
        metadata_lines.append(status_line)

    if trace_path:
        metadata_lines.append(f"**Trace File:** {_format_local_link(trace_path, output_file)}")

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
        "Write exactly 3 concise paragraphs that synthesize the source material into an executive morning briefing. "
        "Use a neutral, professional, analytical journalistic tone. "
        "Do not include a title, heading, preamble, postamble, bullet points, markdown table, or conversational filler. "
        "Do not repeat the phrase 'Morning Briefing Executive Summary'. "
        "Do not invent facts, dates, metrics, weather figures, or reactions that are not present in the source material. "
        "If a source lacks data, acknowledge only what is available without guessing.\n\n"
        f"{summary_input}"
    )

    try:
        llm = _build_summary_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are an expert data synthesizer and technical writer. Produce polished, objective, "
                        "executive-level briefing prose from raw scraper output."
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
    return f"## {SUMMARY_HEADING}\n\n{_strip_duplicate_summary_heading(summary)}\n"


def _format_telemetry_section(telemetry: dict[str, Any] | None) -> str:
    telemetry = telemetry or {}
    prompt_tokens = int(telemetry.get("prompt_tokens", 0) or 0)
    completion_tokens = int(telemetry.get("completion_tokens", 0) or 0)
    total_tokens = int(telemetry.get("total_tokens", 0) or 0)
    execution_time_seconds = float(telemetry.get("execution_time_seconds", 0.0) or 0.0)

    return (
        "## Telemetry & Costs\n\n"
        f"- **Total Prompt Tokens:** {prompt_tokens}\n"
        f"- **Total Completion Tokens:** {completion_tokens}\n"
        f"- **Total Combined Tokens:** {total_tokens}\n"
        f"- **Total Execution Time:** {execution_time_seconds:.3f} seconds\n"
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
