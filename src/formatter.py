from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def _build_report_header(timestamp: datetime) -> str:
    return f"# Morning Briefing\n\nGenerated on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"


def _format_result_section(result: dict[str, Any]) -> str:
    name = str(result.get("name", "Unnamed Target")).strip() or "Unnamed Target"
    url = str(result.get("url", "")).strip()
    status = str(result.get("status", "unknown")).strip().lower()
    content = str(result.get("content", "")).strip()

    if status == "success":
        body = content or "No content was returned."
    else:
        body = f"Extraction failed.\n\n{content or 'No additional error details were provided.'}"

    link_line = f"Source: [{url}]({url})" if url else "Source: URL not provided"
    return f"## {name}\n\n{link_line}\n\n{body}\n"


def generate_markdown_report(results: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sections = [_build_report_header(timestamp)]
    for result in results:
        sections.append(_format_result_section(result))

    markdown = "\n".join(sections).strip() + "\n"
    output_file = OUTPUT_DIR / f"briefing_{timestamp.strftime('%Y-%m-%d')}.md"
    output_file.write_text(markdown, encoding="utf-8")

    return output_file
