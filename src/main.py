from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

try:
    from .browser_agent import extract_data
    from .config import Target, load_targets
    from .formatter import generate_markdown_report
except ImportError:
    from browser_agent import extract_data
    from config import Target, load_targets
    from formatter import generate_markdown_report


def _target_to_dict(target: Target) -> dict[str, str]:
    return asdict(target)


def _is_successful_content(content: str) -> bool:
    return bool(content.strip()) and not content.strip().lower().startswith("error extracting data")


async def main() -> None:
    print("Loading briefing targets...")
    targets = load_targets()

    if not targets:
        print("No targets found in targets.json. Nothing to process.")
        return

    results: list[dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        target_dict = _target_to_dict(target)
        print(f"[{index}/{len(targets)}] Processing {target.name}: {target.url}")

        content = await extract_data(target_dict)
        status = "success" if _is_successful_content(content) else "failure"

        results.append(
            {
                "name": target.name,
                "url": target.url,
                "status": status,
                "content": content,
            }
        )

        print(f"[{index}/{len(targets)}] Finished {target.name} with status: {status}")

    report_path = generate_markdown_report(results)
    print(f"Markdown briefing generated: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
