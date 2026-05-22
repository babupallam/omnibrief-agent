from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, overload

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

    def _get_usage(self, response: "LangChainAIMessage") -> ChatInvokeUsage | None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None

        input_token_details = usage.get("input_token_details") or {}
        return ChatInvokeUsage(
            prompt_tokens=usage.get("input_tokens", 0) or 0,
            prompt_cached_tokens=input_token_details.get("cache_read"),
            prompt_cache_creation_tokens=input_token_details.get("cache_creation"),
            prompt_image_tokens=None,
            completion_tokens=usage.get("output_tokens", 0) or 0,
            total_tokens=usage.get("total_tokens", 0) or 0,
        )

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
                    usage=self._get_usage(response),
                )

            if hasattr(self.chat, "with_structured_output"):
                structured_chat = self.chat.with_structured_output(output_format)
                parsed_object = await structured_chat.ainvoke(langchain_messages, **invoke_kwargs)  # type: ignore[arg-type]
                return ChatInvokeCompletion(completion=parsed_object, usage=None)

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
                usage=self._get_usage(response),
            )
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError(
                message=f"LangChain model error: {exc}",
                model=self.name,
            ) from exc
