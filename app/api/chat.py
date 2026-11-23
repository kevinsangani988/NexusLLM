import uuid
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import json
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import _chat_completions, chat_completions
from app.logger import logger
from app.services.router import select_model
from app.core.redis import get_redis
from redis.asyncio import Redis
from app.services.cache import build_cache_key, cache_response, get_cached_response
from app.services.rate_limit import (
    check_rpm_limit,
    check_tpm_limit,
    record_tpm_usage,
)

router = APIRouter()

async def chat(
    model: str | None,
    message: str,
    routing_strategy: str = "auto",
    temperature: float = 0.7,
    stream: bool = False,
    use_cache: bool = False,
    x_api_key: str = "kevin",
    redis: Redis = Depends(get_redis),
):
    reason = "manual"
    if routing_strategy == "auto":
        model_info = select_model(message)
        model = model_info["model"]
        reason = model_info["reason"]

    request = ChatRequest(
        model=model,
        messages=[{"role": "user", "content": message}],
        routing_strategy=routing_strategy,
        reason=reason if routing_strategy == "auto" else "manual",
        temperature=temperature,
        stream=stream,
    )

    cache_key = None
    if stream and use_cache and redis is not None:
        cache_key = await build_cache_key(message, request.model or "", temperature)

    result = await _chat_completions(
        request,
        redis=redis,
        cache_key=cache_key,
        x_api_key="kevin",
    )

    if not stream and isinstance(result, dict) and use_cache and redis is not None:
        choices = result.get("choices", [])
        if choices:
            answer = choices[0].get("message", {}).get("content", "")
            cache_key = await build_cache_key(message, request.model or "", temperature)
            await cache_response(redis, cache_key, answer)
            
    fallback_used = result.get("_fallback_used", False) if isinstance(result, dict) else False
    attempted_model = result.get("_attempted_model", request.model) if isinstance(result, dict) else request.model
    return result, request.model, reason, request.routing_strategy, fallback_used, attempted_model


@router.post("/v1/chat")
async def chat_endpoint(request: ChatRequest, redis: Redis = Depends(get_redis)):
    request_id = str(uuid.uuid4())

    temperature = request.temperature
    use_cache = temperature <= 0.2

    resolved_model = request.model
    reason = request.reason
    if not resolved_model:
        model_info = select_model(request.message or "")
        resolved_model = model_info["model"]
        reason = model_info["reason"]

    cache_key = await build_cache_key(request.message, resolved_model, temperature) if use_cache else None


    if use_cache:
        cached_response = await get_cached_response(redis, cache_key)
        if cached_response:
            logger.info(f"Cache hit for request_id: {request_id}, model: {resolved_model}")
            if not request.stream:
                return {
                    "model": resolved_model,
                    "choices": [{"message": {"role": "assistant", "content": cached_response}}],
                    "cached": True,
                }
            async def cached_sse():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': cached_response}}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_sse(), media_type="text/event-stream")


    if request.message is not None:
        if request.routing_strategy == "manual":
            chat_result = await chat(resolved_model, request.message, "manual", temperature=temperature, stream=request.stream, use_cache=use_cache, redis=redis)
        else:
            chat_result = await chat(resolved_model, request.message, "auto", temperature=temperature, stream=request.stream, use_cache=use_cache, redis=redis)
        if isinstance(chat_result, tuple):
            result, model, reason, routing_strategy, fallback_used, attempted_model = chat_result
        else:
            result = chat_result
            model = resolved_model
            reason = "manual"
            routing_strategy = request.routing_strategy
            fallback_used = False
            attempted_model = resolved_model
    else:
        result = await chat_completions(request)
        fallback_used = result.get("_fallback_used", False) if isinstance(result, dict) else False
        attempted_model = result.get("_attempted_model", request.model) if isinstance(result, dict) else request.model
        model = attempted_model or request.model
        reason = request.reason
        routing_strategy = request.routing_strategy

    if isinstance(result, StreamingResponse):
        
        return result

    if isinstance(result, str):
        assistant_text = result
        usage = {}
    elif isinstance(result, dict):
        assistant_text = result["choices"][0]["message"]["content"]
        usage = result.get("usage") or {}
    else:
        assistant_text = result.choices[0].message.content
        usage = getattr(result, "usage", None) or {}

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
    else:
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

    logger.info(
        f"Request ID: {request_id}, Model: {request.model}, "
        f"input_tokens: {input_tokens}, output_tokens: {output_tokens}"
    )

    return ChatResponse(
        request_id=request_id,
        model=attempted_model or model,
        response=assistant_text,
        routing_strategy=routing_strategy,
        routing_reason = reason,
        fallback_used=fallback_used,
    )