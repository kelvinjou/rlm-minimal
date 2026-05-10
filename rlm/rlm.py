from abc import ABC, abstractmethod
from dataclasses import dataclass

class RLM(ABC):
    @abstractmethod
    def completion(self, context: list[str] | str | dict[str, str], query: str) -> str:
        pass

    @abstractmethod
    def cost_summary(self) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass

@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_miss_input_tokens": self.cache_miss_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_tokens": self.audio_tokens,
        }

@dataclass
class LLMResult:
    content: str
    usage: LLMUsage | None = None
    cost: float | None = None
    cost_source: str | None = None
    upstream_cost: float | None = None
    response_id: str | None = None
    provider: str | None = None
    model: str | None = None
    reasoning_content: str | None = None
    raw_response: object | None = None

    def as_dict(self, include_raw_response: bool = False) -> dict:
        data = {
            "content": self.content,
            "usage": self.usage.as_dict() if self.usage else None,
            "cost": self.cost,
            "cost_source": self.cost_source,
            "upstream_cost": self.upstream_cost,
            "response_id": self.response_id,
            "provider": self.provider,
            "model": self.model,
            "reasoning_content": self.reasoning_content,
        }
        if include_raw_response:
            data["raw_response"] = self.raw_response
        return data
