import json
from time import perf_counter
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from litellm import acompletion
from app.logger import logger
from app.core.config import settings
from app.schemas.chat import ChatRequest

router = APIRouter()


async def stream_response(model: str, messages: list[dict]):
    time_to_first_token = None
    start_time = perf_counter()
    token_latency = []
    token_count = 0
    last_token_time = start_time

    response = await acompletion(
        model=model,
        messages=messages,
        stream=True,
    )

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
    *,
    model: str | None = None,
    message: str | None = None,
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

    if payload.stream:
        return StreamingResponse(
            stream_response(model=resolved_model, messages=formatted_messages),
            media_type="text/event-stream",
        )

    start_time = perf_counter()
    try:
        response = await acompletion(
            model=resolved_model,
            messages=formatted_messages,
            stream=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    duration_ms = (perf_counter() - start_time) * 1000
    logger.info(f"Sync Metrics | Model: {resolved_model} | Total: {duration_ms:.2f}ms")
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return response