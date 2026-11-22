import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import chat_completions
from app.logger import logger

router = APIRouter()


async def chat(model: str, message: str):
    request = ChatRequest(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    result = await chat_completions(request)

    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result["choices"][0]["message"]["content"]
    return result.choices[0].message.content


@router.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    request_id = str(uuid.uuid4())
    if request.message is not None:
        result = await chat(request.model, request.message)
    else:
        result = await chat_completions(request)

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
        model=request.model,
        response=assistant_text,
    )