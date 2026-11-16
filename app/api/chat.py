from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import completion
import uuid
from app.logger import logger

router = APIRouter()

@router.post("/v1/chat")
async def chat(request: ChatRequest):
    response = await completion(request.model, request.message)
    logger.info(f"Request ID: {str(uuid.uuid4())}, Model: {request.model}, input_token: {response.usage.prompt_tokens}, output_token: {response.usage.completion_tokens}")
    return ChatResponse(
        request_id=str(uuid.uuid4()),  
        model=request.model,
        response=response.choices[0].message.content
    )