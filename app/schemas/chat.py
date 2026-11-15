from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str
    message: str


class ChatResponse(BaseModel):
    request_id: str
    model: str
    response: str