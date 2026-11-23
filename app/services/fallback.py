from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


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
) -> CompletionResult:
    try:
        response = await completion(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
        )
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
        )
        return CompletionResult(
            response=response,
            fallback_used=True,
            attempted_model=fallback_model,
        )
    except Exception as fallback_error:
        raise fallback_error