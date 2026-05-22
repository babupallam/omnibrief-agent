from __future__ import annotations

import asyncio
import re
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
        BASE_URL,
        HEADLESS_MODE,
        LLM_MAX_TOKENS,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_TIMEOUT,
    )
    from .langchain_adapter import ChatLangChain
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
    from langchain_adapter import ChatLangChain


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
    required_keys = {"name", "url", "extraction_prompt"}
    missing_keys = required_keys - target.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        return f"Error extracting data: target is missing required keys: {missing}"

    if not API_KEY:
        return "Error extracting data: API_KEY is not set."

    browser = Browser(headless=HEADLESS_MODE)
    llm = _build_llm()

    try:
        history = await _run_agent_for_target(browser, llm, _build_task(target))
        result_text = _extract_result_text(history, target["name"])

        if history.is_validated() is False or _looks_like_generic_result(result_text, target):
            retry_history = await _run_agent_for_target(browser, llm, _build_task(target, retry=True))
            retry_result_text = _extract_result_text(retry_history, target["name"])

            if retry_result_text and not _looks_like_generic_result(retry_result_text, target):
                return retry_result_text

            if retry_history.is_validated() is True and retry_result_text:
                return retry_result_text

            if result_text:
                return result_text

            return retry_result_text

        return result_text
    except Exception as exc:
        return f"Error extracting data from {target['name']} ({target['url']}): {exc}"
    finally:
        await _close_browser(browser)
