from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from redis.asyncio import Redis
from app.services.rate_limit import record_tpm_usage


@dataclass
class CompletionResult:
    response: Any
    fallback_used: bool
    attempted_model: str


async def complete_with_fallback(
    completion: Callable[..., Awaitable[Any]],
    *,
    model: str,
    fallback_model: str | None,
    messages: list[dict],
    stream: bool,
    temperature: float = 0.7,
    redis: Redis | None = None,
    x_api_key: str = "kevin",
    stream_options: dict | None = None,
) -> CompletionResult:
    try:
        response = await completion(
            model=model,
            messages=messages,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
        )
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        if redis is not None:
            await record_tpm_usage(redis, x_api_key, total_tokens)
        return CompletionResult(
            response=response,
            fallback_used=False,
            attempted_model=model,
        )
    except Exception as primary_error:
        if not fallback_model or fallback_model == model:
            raise primary_error

    try:
        response = await completion(
            model=fallback_model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            stream_options=stream_options,
        )
        return CompletionResult(
            response=response,
            fallback_used=True,
            attempted_model=fallback_model,
        )
    except Exception as fallback_error:
        raise fallback_error