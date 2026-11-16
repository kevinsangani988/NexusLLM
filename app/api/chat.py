import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import chat_completions
from app.logger import logger

router = APIRouter()


@router.post("/v1/chat")
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    result = await chat_completions(request)

    # If streaming, chat_completions returned a StreamingResponse directly
    if isinstance(result, StreamingResponse):
        return result

    # Extract text content and token counts from the dictionary
    assistant_text = result["choices"][0]["message"]["content"]
    usage = result.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    logger.info(
        f"Request ID: {request_id}, Model: {request.model}, "
        f"input_tokens: {input_tokens}, output_tokens: {output_tokens}"
    )

    return ChatResponse(
        request_id=request_id,
        model=request.model,
        response=assistant_text,
    )