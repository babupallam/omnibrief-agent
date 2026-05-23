from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from browser_use import Agent, Browser
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'browser_use' package is not available in the current Python interpreter. "
        "Activate the project virtual environment with 'source .venv/bin/activate' "
        "or run the script with '.venv/bin/python src/browser_agent.py'."
    ) from exc

try:
    from langchain_openai import ChatOpenAI as LangChainChatOpenAI
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'langchain_openai' package is not available in the current Python interpreter. "
        "Activate the project virtual environment with 'source .venv/bin/activate' "
        "or run the script with '.venv/bin/python src/browser_agent.py'."
    ) from exc

try:
    from .config import (
        API_KEY,
        BASE_DIR,
        BASE_URL,
        HEADLESS_MODE,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_TIMEOUT,
    )
    from .langchain_adapter import ChatLangChain
    from .telemetry import TokenUsage, _to_jsonable
except ImportError:
    from config import (
        API_KEY,
        BASE_DIR,
        BASE_URL,
        HEADLESS_MODE,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_TIMEOUT,
    )
    from langchain_adapter import ChatLangChain
    from telemetry import TokenUsage, _to_jsonable


logger = logging.getLogger(__name__)
TRACES_DIR = BASE_DIR / "traces"


@dataclass
class ExtractionResult:
    name: str
    url: str
    status: str
    content: str
    token_usage: TokenUsage
    execution_time_seconds: float
    trace_path: Path | None = None


def _build_task(target: dict[str, str], retry: bool = False) -> str:
    base_instructions = [
        f"You are collecting a morning briefing update for '{target['name']}'.",
        f"Navigate to {target['url']}.",
        f"Complete this extraction task: {target['extraction_prompt']}",
        "Read the actual visible page content before answering.",
        "Return only the extracted result in clean, readable text.",
        "Do not return the site name, section name, page title, domain, or generic labels like 'Homepage' unless that exact text is truly the requested content.",
    ]

    if retry:
        base_instructions.extend(
            [
                "Your previous answer was too generic.",
                "You must extract concrete page content from the DOM or visible page text.",
                "If the page is a homepage, prefer the most prominent content headline rather than branding.",
                "If you cannot find the requested content, say exactly what was missing instead of guessing.",
            ]
        )

    return " ".join(base_instructions)


def _extract_result_text(history: Any, target_name: str) -> str:
    result = history.final_result()
    if result:
        return str(result).strip()

    extracted_content = history.extracted_content()
    if extracted_content:
        if isinstance(extracted_content, list):
            return "\n".join(str(item).strip() for item in extracted_content if item).strip()
        return str(extracted_content).strip()

    return f"Error extracting data: no result was returned for {target_name}."


def _looks_like_generic_result(result: str, target: dict[str, str]) -> bool:
    cleaned_result = result.strip()
    if not cleaned_result:
        return True

    normalized_result = re.sub(r"[^a-z0-9]+", " ", cleaned_result.lower()).strip()
    normalized_name = re.sub(r"[^a-z0-9]+", " ", target["name"].lower()).strip()

    hostname = urlparse(target["url"]).netloc.lower()
    hostname = hostname.removeprefix("www.")
    hostname_label = re.sub(r"[^a-z0-9]+", " ", hostname).strip()

    generic_phrases = {
        "homepage",
        "home page",
        "website",
        "site home",
        "main page",
    }

    if normalized_result in generic_phrases:
        return True
    if normalized_name and normalized_result == normalized_name:
        return True
    if hostname_label and normalized_result == hostname_label:
        return True
    if "homepage" in normalized_result and len(normalized_result.split()) <= 3:
        return True

    return False


async def _run_agent_for_target(browser: Browser, llm: ChatLangChain, task: str) -> Any:
    agent = Agent(
        task=task,
        browser=browser,
        llm=llm,
        use_judge=True,
    )
    return await agent.run()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return slug or "target"


def _serialize_history(history: Any) -> dict[str, Any] | None:
    if history is None:
        return None
    if hasattr(history, "model_dump"):
        return history.model_dump()
    return _to_jsonable(history)


def _save_trace(
    *,
    target: dict[str, str],
    attempts: list[dict[str, Any]],
    telemetry: dict[str, Any],
    status: str,
    content: str,
    execution_time_seconds: float,
    error: str | None = None,
) -> Path:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = TRACES_DIR / f"trace_{_slugify(target.get('name', 'target'))}_{timestamp}.json"

    trace_data = {
        "target": {
            "name": target.get("name"),
            "url": target.get("url"),
            "extraction_prompt": target.get("extraction_prompt"),
        },
        "status": status,
        "content": content,
        "error": error,
        "execution_time_seconds": round(execution_time_seconds, 3),
        "telemetry": telemetry,
        "attempts": attempts,
    }

    trace_path.write_text(json.dumps(_to_jsonable(trace_data), indent=2), encoding="utf-8")
    logger.info("Saved execution trace to %s", trace_path)
    return trace_path


def _empty_telemetry() -> dict[str, Any]:
    return {
        "total_usage": TokenUsage().to_dict(),
        "llm_calls": [],
    }


def _build_llm() -> ChatLangChain:
    llm_kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "api_key": API_KEY,
    }

    if BASE_URL:
        llm_kwargs["base_url"] = BASE_URL
    if LLM_TEMPERATURE is not None:
        llm_kwargs["temperature"] = LLM_TEMPERATURE
    if LLM_MAX_TOKENS is not None:
        llm_kwargs["max_tokens"] = LLM_MAX_TOKENS
    if LLM_TIMEOUT is not None:
        llm_kwargs["timeout"] = LLM_TIMEOUT

    langchain_model = LangChainChatOpenAI(**llm_kwargs)
    return ChatLangChain(chat=langchain_model)


async def _close_browser(browser: Browser) -> None:
    for method_name in ("stop", "close", "kill"):
        method = getattr(browser, method_name, None)
        if callable(method):
            result = method()
            if asyncio.iscoroutine(result):
                await result
            return


async def extract_data(target: dict[str, str]) -> str:
    result = await extract_data_with_trace(target)
    return result.content


async def extract_data_with_trace(target: dict[str, str]) -> ExtractionResult:
    start_time = time.perf_counter()
    required_keys = {"name", "url", "extraction_prompt"}
    missing_keys = required_keys - target.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        content = f"Error extracting data: target is missing required keys: {missing}"
        execution_time = time.perf_counter() - start_time
        trace_path = _save_trace(
            target=target,
            attempts=[],
            telemetry=_empty_telemetry(),
            status="failure",
            content=content,
            execution_time_seconds=execution_time,
            error=content,
        )
        return ExtractionResult(
            name=target.get("name", "Unknown"),
            url=target.get("url", ""),
            status="failure",
            content=content,
            token_usage=TokenUsage(),
            execution_time_seconds=execution_time,
            trace_path=trace_path,
        )

    if not API_KEY:
        content = "Error extracting data: API_KEY is not set."
        execution_time = time.perf_counter() - start_time
        trace_path = _save_trace(
            target=target,
            attempts=[],
            telemetry=_empty_telemetry(),
            status="failure",
            content=content,
            execution_time_seconds=execution_time,
            error=content,
        )
        return ExtractionResult(
            name=target["name"],
            url=target["url"],
            status="failure",
            content=content,
            token_usage=TokenUsage(),
            execution_time_seconds=execution_time,
            trace_path=trace_path,
        )

    browser = Browser(headless=HEADLESS_MODE)
    llm = _build_llm()
    attempts: list[dict[str, Any]] = []
    final_content = ""
    final_status = "failure"

    try:
        history = await _run_agent_for_target(browser, llm, _build_task(target))
        result_text = _extract_result_text(history, target["name"])
        attempts.append(
            {
                "attempt": 1,
                "task": _build_task(target),
                "validated": history.is_validated(),
                "final_result": result_text,
                "history": _serialize_history(history),
            }
        )

        if history.is_validated() is False or _looks_like_generic_result(result_text, target):
            retry_history = await _run_agent_for_target(browser, llm, _build_task(target, retry=True))
            retry_result_text = _extract_result_text(retry_history, target["name"])
            attempts.append(
                {
                    "attempt": 2,
                    "task": _build_task(target, retry=True),
                    "validated": retry_history.is_validated(),
                    "final_result": retry_result_text,
                    "history": _serialize_history(retry_history),
                }
            )

            if retry_result_text and not _looks_like_generic_result(retry_result_text, target):
                final_content = retry_result_text
                final_status = "success"
            elif retry_history.is_validated() is True and retry_result_text:
                final_content = retry_result_text
                final_status = "success"
            elif result_text:
                final_content = result_text
                final_status = "failure"
            else:
                final_content = retry_result_text
                final_status = "failure"
        else:
            final_content = result_text
            final_status = "success"

        execution_time = time.perf_counter() - start_time
        trace_path = _save_trace(
            target=target,
            attempts=attempts,
            telemetry=llm.telemetry.to_dict(),
            status=final_status,
            content=final_content,
            execution_time_seconds=execution_time,
        )

        return ExtractionResult(
            name=target["name"],
            url=target["url"],
            status=final_status,
            content=final_content,
            token_usage=llm.telemetry.total_usage,
            execution_time_seconds=execution_time,
            trace_path=trace_path,
        )
    except Exception as exc:
        execution_time = time.perf_counter() - start_time
        final_content = f"Error extracting data from {target['name']} ({target['url']}): {exc}"
        logger.exception("Extraction failed for %s", target["name"])
        trace_path = _save_trace(
            target=target,
            attempts=attempts,
            telemetry=llm.telemetry.to_dict(),
            status="failure",
            content=final_content,
            execution_time_seconds=execution_time,
            error=str(exc),
        )
        return ExtractionResult(
            name=target["name"],
            url=target["url"],
            status="failure",
            content=final_content,
            token_usage=llm.telemetry.total_usage,
            execution_time_seconds=execution_time,
            trace_path=trace_path,
        )
    finally:
        await _close_browser(browser)
