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
from app.services.cache import build_cache_key
from app.services.fallback import complete_with_fallback

router = APIRouter()


async def stream_response(model: str, messages: list[dict]):
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
    )
    response = result.response

    async for chunk in response:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", "")

        if content:
            now = perf_counter()
            if time_to_first_token is None:
                time_to_first_token = (now - start_time) * 1000
            else:
                token_latency.append((now - last_token_time) * 1000)

            token_count += 1
            last_token_time = now
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    yield "data: [DONE]\n\n"

    total_duration_ms = (perf_counter() - start_time) * 1000
    avg_itl = sum(token_latency) / len(token_latency) if token_latency else 0.0
    ttft_display = f"{time_to_first_token:.2f}ms" if time_to_first_token is not None else "N/A"

    logger.info(
        f"Stream Metrics | Model: {model} | Tokens: {token_count} | "
        f"TTFT: {ttft_display} | Avg ITL: {avg_itl:.2f}ms | "
        f"Total: {total_duration_ms:.2f}ms"
    )
    

@router.post("/chat/completions")
async def chat_completions(
    payload: ChatRequest | None = None,
    redis: Redis = Depends(get_redis)
):
    if payload is None:
        if payload is None:
            raise HTTPException(
                status_code=422,
                detail="model and message are required",
            )
        return {"error": "model and message are required"}

    resolved_model = settings.model_registry.get(payload.model, payload.model)
    formatted_messages = [m.model_dump() for m in payload.messages]
    if payload.message is not None and not formatted_messages:
        formatted_messages = [{"role": "user", "content": payload.message}]

    use_cache = payload.temperature <= 0.2
    cache_key = build_cache_key(payload.message, resolved_model, payload.temperature) 
    

    if payload.stream:
        return StreamingResponse(
            stream_response(model=resolved_model, messages=formatted_messages),
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