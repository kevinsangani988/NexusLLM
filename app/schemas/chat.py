from typing import Literal
from pydantic import BaseModel, Field
from pydantic import model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    message: str | None = None
    stream: bool = False

    @model_validator(mode="after")
    def require_message(self):
        if not self.message and not self.messages:
            raise ValueError("message or messages is required")
        return self


class ChatResponse(BaseModel):
    request_id: str
    model: str
    response: str