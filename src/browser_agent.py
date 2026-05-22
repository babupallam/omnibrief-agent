from __future__ import annotations

import asyncio
from typing import Any

from browser_use import Agent, Browser
from langchain_openai import ChatOpenAI

try:
    from .config import (
        API_KEY,
        BASE_URL,
        HEADLESS_MODE,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_TIMEOUT,
    )
except ImportError:
    from config import (
        API_KEY,
        BASE_URL,
        HEADLESS_MODE,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_TIMEOUT,
    )


def _build_llm() -> ChatOpenAI:
    llm_kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "api_key": API_KEY,
        "temperature": LLM_TEMPERATURE,
    }

    if BASE_URL:
        llm_kwargs["base_url"] = BASE_URL
    if LLM_MAX_TOKENS is not None:
        llm_kwargs["max_tokens"] = LLM_MAX_TOKENS
    if LLM_TIMEOUT is not None:
        llm_kwargs["timeout"] = LLM_TIMEOUT

    return ChatOpenAI(**llm_kwargs)


async def _close_browser(browser: Browser) -> None:
    for method_name in ("stop", "close", "kill"):
        method = getattr(browser, method_name, None)
        if callable(method):
            result = method()
            if asyncio.iscoroutine(result):
                await result
            return


async def extract_data(target: dict[str, str]) -> str:
    required_keys = {"name", "url", "extraction_prompt"}
    missing_keys = required_keys - target.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        return f"Error extracting data: target is missing required keys: {missing}"

    if not API_KEY:
        return "Error extracting data: API_KEY is not set."

    browser = Browser(headless=HEADLESS_MODE)
    llm = _build_llm()
    task = (
        f"You are collecting a morning briefing update for '{target['name']}'. "
        f"Navigate to {target['url']} and complete this extraction task: "
        f"{target['extraction_prompt']} "
        "Return only the extracted result in clean, readable text."
    )

    try:
        agent = Agent(
            task=task,
            browser=browser,
            llm=llm,
        )
        history = await agent.run()

        result = history.final_result()
        if result:
            return str(result).strip()

        extracted_content = history.extracted_content()
        if extracted_content:
            return "\n".join(str(item).strip() for item in extracted_content if item).strip()

        return f"Error extracting data: no result was returned for {target['name']}."
    except Exception as exc:
        return f"Error extracting data from {target['name']} ({target['url']}): {exc}"
    finally:
        await _close_browser(browser)
