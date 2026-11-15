from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import completion
import uuid

router = APIRouter()

@router.post("v1/chat")
async def chat(request: ChatRequest):
    response = await completion(request.model, request.message)
    return ChatResponse(
        request_id=str(uuid.uuid4()),  
        model=request.model,
        response=response
    )