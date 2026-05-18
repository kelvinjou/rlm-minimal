"""
LLM client wrappers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from dotenv import load_dotenv

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
    "nvidia": OpenAICompatibleProvider(
        api_key_env="NVIDIA_API_KEY",
        # https://integrate.api.nvidia.com/v1/chat/completions # for kimi2.6
        # https://integrate.api.nvidia.com/v1 # for deepseek v4 pro
        base_url="https://integrate.api.nvidia.com/v1",
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
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        # For keeping track of token usage 
        self.last_usage = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

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
        self.timeout = timeout
        self.max_retries = max_retries
        base_url_env = f"{provider.upper()}_BASE_URL"
        default_base_url = provider_defaults.base_url if provider_defaults else None
        self.base_url = base_url or os.getenv(base_url_env) or os.getenv("LLM_BASE_URL") or default_base_url
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429, 500, 502, 503, 504}
        return False
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            request_params = {
                "model": self.model,
                "messages": messages,
                "timeout": self.timeout,
                **kwargs,
            }

            if max_tokens is not None:
                # OpenRouter and most OpenAI-compatible chat endpoints expect
                # max_tokens. Avoid sending max_completion_tokens=None.
                request_params["max_tokens"] = max_tokens

            response = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(**request_params)
                    self._record_usage(response)
                    break
                except Exception as e:
                    if attempt >= self.max_retries or not self._is_retryable_error(e):
                        raise

                    delay = min(2 ** attempt, 8)
                    status = getattr(e, "status_code", None)
                    error_label = f"status {status}" if status else type(e).__name__
                    print(
                        f"LLM request failed with {error_label}; "
                        f"retrying in {delay}s ({attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(delay)

            if response is None:
                raise RuntimeError("LLM request failed without returning a response.")

            choices = getattr(response, "choices", None)
            if not choices:
                response_error = getattr(response, "error", None)
                if response_error:
                    raise RuntimeError(f"API response error: {response_error}")

                response_dump = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else repr(response)
                )
                raise RuntimeError(f"API response did not include choices: {response_dump}")

            message = choices[0].message
            content = getattr(message, "content", None)
            if content is None:
                response_dump = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else repr(response)
                )
                raise RuntimeError(f"API response message did not include content: {response_dump}")

            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            return content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
        
    def _record_usage(self, response) -> None:
        if self.provider != "nvidia":
            return
        
        usage = getattr(response, "usage", None)
        self.last_usage = usage

        if usage is None:
            return
        
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens


def create_llm_client(
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    timeout: float = 300.0,
    max_retries: int = 3,
) -> OpenAICompatibleClient:
    provider = provider or os.getenv("RLM_CLIENT_BACKEND", "openai")
    return OpenAICompatibleClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        timeout=timeout,
        max_retries=max_retries,
    )


class OpenAIClient(OpenAICompatibleClient):
    """Backward-compatible alias for the OpenAI-compatible client."""
