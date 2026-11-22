import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import chat_completions
from app.logger import logger
from app.services.router import select_model

router = APIRouter()

async def chat(model: str | None, message: str, routing_strategy: str = "auto"):
    reason = "manual routing"
    if routing_strategy == "auto":
        model_info = select_model(message)
        model = model_info["model"]
        reason = model_info["reason"]

    request = ChatRequest(
        model=model,
        messages=[{"role": "user", "content": message}],
        routing_strategy=routing_strategy,
        reason=reason if routing_strategy == "auto" else "manual"
    )
    result = await chat_completions(request)
    fallback_used = result.get("_fallback_used", False) if isinstance(result, dict) else False
    attempted_model = result.get("_attempted_model", request.model) if isinstance(result, dict) else request.model
    return result, request.model, reason, request.routing_strategy, fallback_used, attempted_model


@router.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    request_id = str(uuid.uuid4())
    if request.message is not None:
        if request.routing_strategy == "manual":
            chat_result = await chat(request.model, request.message, "manual")
        else:
            chat_result = await chat(request.model, request.message)
        if isinstance(chat_result, tuple):
            result, model, reason, routing_strategy, fallback_used, attempted_model = chat_result
        else:
            result = chat_result
            model = request.model
            reason = "manual"
            routing_strategy = request.routing_strategy
            fallback_used = False
            attempted_model = request.model
    else:
        result = await chat_completions(request)
        fallback_used = result.get("_fallback_used", False) if isinstance(result, dict) else False
        attempted_model = result.get("_attempted_model", request.model) if isinstance(result, dict) else request.model
        model = attempted_model or request.model
        reason = request.reason
        routing_strategy = request.routing_strategy

    # If streaming, chat_completions returned a StreamingResponse directly
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