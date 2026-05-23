from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class LLMCallTrace:
    timestamp: str
    provider: str
    model: str
    request_type: str
    output_format: str | None
    usage: TokenUsage
    raw_response: Any = None
    response_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage"] = self.usage.to_dict()
        return data


class TelemetryRecorder:
    def __init__(self) -> None:
        self.calls: list[LLMCallTrace] = []
        self.total_usage = TokenUsage()

    def record_call(
        self,
        *,
        provider: str,
        model: str,
        request_type: str,
        output_format: str | None,
        usage: TokenUsage,
        raw_response: Any = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.total_usage.add(usage)
        self.calls.append(
            LLMCallTrace(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                provider=provider,
                model=model,
                request_type=request_type,
                output_format=output_format,
                usage=usage,
                raw_response=_to_jsonable(raw_response),
                response_metadata=_to_jsonable(response_metadata or {}),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usage": self.total_usage.to_dict(),
            "llm_calls": [call.to_dict() for call in self.calls],
        }


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, TokenUsage):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
