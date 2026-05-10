"""
LLM client wrappers.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from rlm.helper import DEEPSEEK_PRICING_PER_1M, _get_nested, _to_dict, _to_float, _to_int
from rlm.rlm import LLMResult, LLMUsage

load_dotenv()

@dataclass(frozen=True)
class OpenAICompatibleProvider:
    api_key_env: str
    base_url: Optional[str]


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai": OpenAICompatibleProvider(
        api_key_env="OPENAI_API_KEY",
        base_url=None,
    ),
    "openrouter": OpenAICompatibleProvider(
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "deepseek": OpenAICompatibleProvider(
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "openai_compatible": OpenAICompatibleProvider(
        api_key_env="LLM_API_KEY",
        base_url=None,
    ),
}


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        base_url: Optional[str] = None,
        provider: str = "openai",
        call_history: Optional[list[LLMResult]] = None,
    ):
        provider_defaults = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if provider_defaults is None and base_url is None:
            valid_providers = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
            raise ValueError(
                f"Unknown provider '{provider}'. Use one of: {valid_providers}, "
                "or pass base_url for a custom OpenAI-compatible endpoint."
            )

        api_key_env = provider_defaults.api_key_env if provider_defaults else "LLM_API_KEY"
        self.api_key = api_key or os.getenv(api_key_env) or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                f"API key is required for provider '{provider}'. Set {api_key_env} "
                "or LLM_API_KEY, or pass api_key."
            )
        
        self.provider = provider
        self.model = model
        base_url_env = f"{provider.upper()}_BASE_URL"
        default_base_url = provider_defaults.base_url if provider_defaults else None
        self.base_url = base_url or os.getenv(base_url_env) or os.getenv("LLM_BASE_URL") or default_base_url
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.call_history = call_history if call_history is not None else []

    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        return self.completion_with_metadata(
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        ).content

    def completion_with_metadata(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            request_params = {
                "model": self.model,
                "messages": messages,
                **kwargs,
            }

            if max_tokens is not None:
                # OpenRouter and most OpenAI-compatible chat endpoints expect
                # max_tokens. Avoid sending max_completion_tokens=None.
                request_params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**request_params)
            result = self._parse_response(response)
            self.call_history.append(result)
            return result

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    def _parse_response(self, response) -> LLMResult:
        response_data = _to_dict(response)
        choices = response_data.get("choices") or []
        if not choices:
            response_error = response_data.get("error")
            if response_error:
                raise RuntimeError(f"API response error: {response_error}")

            raise RuntimeError(f"API response did not include choices: {response_data or repr(response)}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"API response message did not include content: {response_data or repr(response)}")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        usage = self._parse_usage(response_data.get("usage") or {})
        cost, cost_source, upstream_cost = self._parse_cost(response_data, usage)

        return LLMResult(
            content=content,
            usage=usage,
            cost=cost,
            cost_source=cost_source,
            upstream_cost=upstream_cost,
            response_id=response_data.get("id"),
            provider=self.provider,
            model=response_data.get("model") or self.model,
            reasoning_content=message.get("reasoning_content"),
            raw_response=response,
        )

    def _parse_usage(self, usage_data: dict) -> Optional[LLMUsage]:
        if not usage_data:
            return None

        prompt_details = usage_data.get("prompt_tokens_details") or {}
        completion_details = usage_data.get("completion_tokens_details") or {}

        cached_input_tokens = (
            _to_int(usage_data.get("prompt_cache_hit_tokens"))
            or _to_int(prompt_details.get("cached_tokens"))
        )
        cache_miss_input_tokens = _to_int(usage_data.get("prompt_cache_miss_tokens"))

        return LLMUsage(
            input_tokens=_to_int(usage_data.get("prompt_tokens")),
            output_tokens=_to_int(usage_data.get("completion_tokens")),
            total_tokens=_to_int(usage_data.get("total_tokens")),
            cached_input_tokens=cached_input_tokens,
            cache_miss_input_tokens=cache_miss_input_tokens,
            cache_write_tokens=_to_int(prompt_details.get("cache_write_tokens")),
            reasoning_tokens=_to_int(completion_details.get("reasoning_tokens")),
            audio_tokens=(
                _to_int(prompt_details.get("audio_tokens"))
                + _to_int(completion_details.get("audio_tokens"))
            ),
        )

    def _parse_cost(
        self,
        response_data: dict,
        usage: Optional[LLMUsage],
    ) -> tuple[Optional[float], Optional[str], Optional[float]]:
        usage_data = response_data.get("usage") or {}

        provider_cost = _to_float(usage_data.get("cost"))
        upstream_cost = _to_float(_get_nested(usage_data, "cost_details", "upstream_inference_cost"))
        if provider_cost is not None:
            return provider_cost, "provider", upstream_cost

        if self.provider == "deepseek" and usage:
            estimated_cost = self._estimate_deepseek_cost(usage)
            if estimated_cost is not None:
                return estimated_cost, "estimated", upstream_cost

        return None, None, upstream_cost

    def _estimate_deepseek_cost(self, usage: LLMUsage) -> Optional[float]:
        pricing = DEEPSEEK_PRICING_PER_1M.get(self.model)
        if not pricing:
            return None

        cache_miss_input_tokens = usage.cache_miss_input_tokens
        if cache_miss_input_tokens == 0 and usage.input_tokens:
            cache_miss_input_tokens = max(0, usage.input_tokens - usage.cached_input_tokens)

        return (
            usage.cached_input_tokens * pricing["input_cache_hit"]
            + cache_miss_input_tokens * pricing["input_cache_miss"]
            + usage.output_tokens * pricing["output"]
        ) / 1_000_000


def create_llm_client(
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    call_history: Optional[list[LLMResult]] = None,
) -> OpenAICompatibleClient:
    provider = provider or os.getenv("RLM_CLIENT_BACKEND", "openai")
    return OpenAICompatibleClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        call_history=call_history,
    )


class OpenAIClient(OpenAICompatibleClient):
    """Backward-compatible alias for the OpenAI-compatible client."""
