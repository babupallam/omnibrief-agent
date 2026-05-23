from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    UserMessage,
)
from browser_use.llm.messages import SystemMessage as BrowserUseSystemMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage as LangChainBaseMessage
from pydantic import BaseModel

try:
    from .config import AGENT_USE_VISION
    from .telemetry import TelemetryRecorder, TokenUsage
except ImportError:
    from config import AGENT_USE_VISION
    from telemetry import TelemetryRecorder, TokenUsage

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel
    from langchain_core.messages import AIMessage as LangChainAIMessage


T = TypeVar("T", bound=BaseModel)


class LangChainMessageSerializer:
    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> str | list[str | dict]:
        if isinstance(content, str):
            return content

        if not AGENT_USE_VISION:
            text_parts: list[str] = []
            for part in content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "image_url":
                    image_url = part.image_url.url
                    if not image_url.startswith("data:image"):
                        text_parts.append(f"[Image omitted for text-only model: {image_url}]")
            return "\n".join(text_parts).strip()

        serialized_parts: list[str | dict] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append({"type": "text", "text": part.text})
            elif part.type == "image_url":
                serialized_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": part.image_url.url,
                            "detail": part.image_url.detail,
                        },
                    }
                )
        return serialized_parts

    @staticmethod
    def _serialize_system_content(content: str | list[ContentPartTextParam]) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if part.type == "text")

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
    ) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if part.type == "text")

    @staticmethod
    def serialize(message: BaseMessage) -> LangChainBaseMessage:
        if isinstance(message, UserMessage):
            return HumanMessage(
                content=LangChainMessageSerializer._serialize_user_content(message.content),
                name=message.name,
            )

        if isinstance(message, BrowserUseSystemMessage):
            return SystemMessage(
                content=LangChainMessageSerializer._serialize_system_content(message.content),
                name=message.name,
            )

        if isinstance(message, AssistantMessage):
            return AIMessage(
                content=LangChainMessageSerializer._serialize_assistant_content(message.content),
                name=message.name,
            )

        raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(messages: list[BaseMessage]) -> list[LangChainBaseMessage]:
        return [LangChainMessageSerializer.serialize(message) for message in messages]


@dataclass
class ChatLangChain(BaseChatModel):
    chat: "LangChainBaseChatModel"
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)

    @staticmethod
    def _build_langchain_invoke_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
        invoke_kwargs: dict[str, object] = {}

        # Browser Use may pass session-scoped metadata that LangChain chat models do not accept.
        if "config" in kwargs and kwargs["config"] is not None:
            invoke_kwargs["config"] = kwargs["config"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            invoke_kwargs["stop"] = kwargs["stop"]

        return invoke_kwargs

    @property
    def model(self) -> str:
        return self.name

    @property
    def provider(self) -> str:
        model_class_name = self.chat.__class__.__name__.lower()
        if "openai" in model_class_name:
            return "openai"
        if "anthropic" in model_class_name or "claude" in model_class_name:
            return "anthropic"
        if "google" in model_class_name or "gemini" in model_class_name:
            return "google"
        if "groq" in model_class_name:
            return "groq"
        if "ollama" in model_class_name:
            return "ollama"
        if "deepseek" in model_class_name:
            return "deepseek"
        return "langchain"

    @property
    def name(self) -> str:
        model_name = getattr(self.chat, "model_name", None)
        if model_name:
            return str(model_name)

        model_attr = getattr(self.chat, "model", None)
        if model_attr:
            return str(model_attr)

        return self.chat.__class__.__name__

    def _extract_token_usage(self, response: "LangChainAIMessage") -> TokenUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0) or 0,
                completion_tokens=usage.get("output_tokens", 0) or 0,
                total_tokens=usage.get("total_tokens", 0) or 0,
            )

        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        return TokenUsage(
            prompt_tokens=token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0,
            completion_tokens=token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0,
            total_tokens=token_usage.get("total_tokens", 0) or 0,
        )

    def _get_usage(self, response: "LangChainAIMessage") -> ChatInvokeUsage | None:
        usage = getattr(response, "usage_metadata", None)
        input_token_details = usage.get("input_token_details") if usage else {}
        input_token_details = input_token_details or {}
        token_usage = self._extract_token_usage(response)

        if token_usage.total_tokens == 0 and token_usage.prompt_tokens == 0 and token_usage.completion_tokens == 0:
            return None

        return ChatInvokeUsage(
            prompt_tokens=token_usage.prompt_tokens,
            prompt_cached_tokens=input_token_details.get("cache_read"),
            prompt_cache_creation_tokens=input_token_details.get("cache_creation"),
            prompt_image_tokens=None,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
        )

    def _record_response_usage(
        self,
        response: "LangChainAIMessage",
        *,
        request_type: str,
        output_format: type[BaseModel] | None,
    ) -> ChatInvokeUsage | None:
        usage = self._get_usage(response)
        token_usage = self._extract_token_usage(response)
        self.telemetry.record_call(
            provider=self.provider,
            model=self.name,
            request_type=request_type,
            output_format=output_format.__name__ if output_format else None,
            usage=token_usage,
            raw_response=getattr(response, "content", None),
            response_metadata=getattr(response, "response_metadata", None) or {},
        )
        return usage

    @overload
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T],
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        langchain_messages = LangChainMessageSerializer.serialize_messages(messages)
        invoke_kwargs = self._build_langchain_invoke_kwargs(kwargs)
        request_type = str(kwargs.get("request_type") or ("structured_output" if output_format else "browser_agent"))

        try:
            if output_format is None:
                response = await self.chat.ainvoke(langchain_messages, **invoke_kwargs)  # type: ignore[arg-type]
                if not isinstance(response, AIMessage):
                    raise ModelProviderError(
                        message=f"Response is not an AIMessage: {type(response)}",
                        model=self.name,
                    )

                return ChatInvokeCompletion(
                    completion=str(response.content),
                    usage=self._record_response_usage(
                        response,
                        request_type=request_type,
                        output_format=None,
                    ),
                )

            if hasattr(self.chat, "with_structured_output"):
                try:
                    structured_chat = self.chat.with_structured_output(output_format, include_raw=True)
                    structured_response = await structured_chat.ainvoke(langchain_messages, **invoke_kwargs)  # type: ignore[arg-type]
                except TypeError:
                    structured_chat = self.chat.with_structured_output(output_format)
                    parsed_object = await structured_chat.ainvoke(langchain_messages, **invoke_kwargs)  # type: ignore[arg-type]
                    self.telemetry.record_call(
                        provider=self.provider,
                        model=self.name,
                        request_type=request_type,
                        output_format=output_format.__name__,
                        usage=TokenUsage(),
                        raw_response=parsed_object,
                        response_metadata={"usage_missing": True},
                    )
                    return ChatInvokeCompletion(completion=parsed_object, usage=None)

                if not isinstance(structured_response, dict):
                    self.telemetry.record_call(
                        provider=self.provider,
                        model=self.name,
                        request_type=request_type,
                        output_format=output_format.__name__,
                        usage=TokenUsage(),
                        raw_response=structured_response,
                        response_metadata={"usage_missing": True},
                    )
                    return ChatInvokeCompletion(completion=structured_response, usage=None)

                raw_response = structured_response.get("raw")
                parsed_object = structured_response.get("parsed")
                parsing_error = structured_response.get("parsing_error")

                if isinstance(raw_response, AIMessage):
                    usage = self._record_response_usage(
                        raw_response,
                        request_type=request_type,
                        output_format=output_format,
                    )
                else:
                    usage = None
                    self.telemetry.record_call(
                        provider=self.provider,
                        model=self.name,
                        request_type=request_type,
                        output_format=output_format.__name__,
                        usage=TokenUsage(),
                        raw_response=raw_response,
                        response_metadata={"usage_missing": True},
                    )

                if parsed_object is None and parsing_error is not None:
                    raise ModelProviderError(
                        message=f"Structured output parsing failed: {parsing_error}",
                        model=self.name,
                    )

                return ChatInvokeCompletion(completion=parsed_object, usage=usage)

            response = await self.chat.ainvoke(langchain_messages, **invoke_kwargs)  # type: ignore[arg-type]
            if not isinstance(response, AIMessage):
                raise ModelProviderError(
                    message=f"Response is not an AIMessage: {type(response)}",
                    model=self.name,
                )

            content = response.content if hasattr(response, "content") else str(response)
            if not isinstance(content, str):
                raise ModelProviderError(
                    message=f"Expected string content for structured fallback, got: {type(content)}",
                    model=self.name,
                )

            parsed_data = json.loads(content)
            if not isinstance(parsed_data, dict):
                raise ModelProviderError(
                    message=f"Structured output is not a JSON object: {type(parsed_data)}",
                    model=self.name,
                )

            return ChatInvokeCompletion(
                completion=output_format(**parsed_data),
                usage=self._record_response_usage(
                    response,
                    request_type=request_type,
                    output_format=output_format,
                ),
            )
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(
                message=f"LangChain model error: {exc}",
                model=self.name,
            ) from exc
