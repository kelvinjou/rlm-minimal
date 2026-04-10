from .rlm import RLM

from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "openai_compatible",
    "deepseek",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]
