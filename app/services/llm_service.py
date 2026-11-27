import json
from time import perf_counter
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from litellm import acompletion
from app.logger import logger
from app.core.config import settings
from redis.asyncio import Redis
from app.core.redis import get_redis
from app.schemas.chat import ChatRequest
from app.services.cache import cache_response
from app.services.semantic_cache import store_semantic_cache
from app.services.fallback import complete_with_fallback
from app.services.rate_limit import check_rpm_limit, check_tpm_limit, record_tpm_usage

router = APIRouter()


async def stream_response(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    redis: Redis | None = None,
    cache_key: str | None = None,
    cache_model: str | None = None,
    x_api_key: str = "kevin"
):
    time_to_first_token = None
    start_time = perf_counter()
    token_latency = []
    token_count = 0
    last_token_time = start_time


    fallback_model = settings.fallback_registry.get(model)
    result = await complete_with_fallback(
        acompletion,
        model=model,
        fallback_model=fallback_model,
        messages=messages,
        stream=True,
        temperature=temperature,
        stream_options={"include_usage": True},
        redis=redis,
        x_api_key=x_api_key,
    )
    response = result.response
    chunks: list[str] = []

    async for chunk in response:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage and getattr(chunk_usage, "total_tokens", None):
            total_tokens = chunk_usage.total_tokens
            if redis is not None:
                await record_tpm_usage(redis, x_api_key, total_tokens)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None

        if content:
            chunks.append(content)
            now = perf_counter()
            if time_to_first_token is None:
                time_to_first_token = (now - start_time) * 1000
            else:
                token_latency.append((now - last_token_time) * 1000)

            token_count += 1
            last_token_time = now
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    yield "data: [DONE]\n\n"

    if redis is not None and cache_key is not None and chunks:
        full_response = "".join(chunks)
        await cache_response(redis, cache_key, full_response)
        await store_semantic_cache(cache_model or model, messages, full_response)

    total_duration_ms = (perf_counter() - start_time) * 1000
    avg_itl = sum(token_latency) / len(token_latency) if token_latency else 0.0
    ttft_display = f"{time_to_first_token:.2f}ms" if time_to_first_token is not None else "N/A"

    logger.info(
        f"Stream Metrics | Model: {model} | Tokens: {token_count} | "
        f"TTFT: {ttft_display} | Avg ITL: {avg_itl:.2f}ms | "
        f"Total: {total_duration_ms:.2f}ms"
    )
    

async def _chat_completions(
    payload: ChatRequest | None = None,
    *,
    model: str | None = None,
    message: str | None = None,
    redis: Redis | None = None,
    cache_key: str | None = None,
    x_api_key: str = "kevin" ,
):
    if payload is None:
        if model is None or message is None:
            raise HTTPException(
                status_code=422,
                detail="model and message are required",
            )
        payload = ChatRequest(
            model=model,
            messages=[{"role": "user", "content": message}],
        )

    resolved_model = settings.model_registry.get(payload.model, payload.model)
    formatted_messages = [m.model_dump() for m in payload.messages]
    if payload.message is not None and not formatted_messages:
        formatted_messages = [{"role": "user", "content": payload.message}]

    if redis is not None:
        await check_rpm_limit(redis, x_api_key, rpm_limit=30)
        await check_tpm_limit(redis, x_api_key, tpm_limit=100_000)

    if payload.stream:
        return StreamingResponse(
            stream_response(
                model=resolved_model,
                messages=formatted_messages,
                temperature=payload.temperature,
                redis=redis,
                cache_key=cache_key,
                cache_model=payload.model or resolved_model,
            ),
            media_type="text/event-stream",
        )

    start_time = perf_counter()
    try:
        fallback_model = settings.fallback_registry.get(resolved_model)
        result = await complete_with_fallback(
            acompletion,
            model=resolved_model,
            fallback_model=fallback_model,
            messages=formatted_messages,
            temperature=payload.temperature,
            stream=False,
            redis=redis,
            x_api_key=x_api_key,
        )
        response = result.response
        fallback_used = result.fallback_used
        attempted_model = result.attempted_model
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(f"Sync Metrics | Model: {resolved_model} | Total: {duration_ms:.2f}ms")
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
        payload["_fallback_used"] = fallback_used
        payload["_attempted_model"] = attempted_model
        return payload
    if isinstance(response, dict):
        response["_fallback_used"] = fallback_used
        response["_attempted_model"] = attempted_model
        return response
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return response


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatRequest | None = None,
    *,
    model: str | None = None,
    message: str | None = None,
    api_key: str = "kevin",
):
    return await _chat_completions(
        payload,
        model=model,
        message=message,
    )