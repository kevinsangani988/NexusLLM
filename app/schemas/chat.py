from typing import Literal
from pydantic import BaseModel, Field
from pydantic import model_validator
from typing import Optional


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    message: str | None = None
    stream: bool = False
    routing_strategy: Literal["auto", "manual"] = "auto"
    reason: Optional[str] = None
    temperature: float = 0.7

    @model_validator(mode="after")
    def require_message(self):
        if not self.message and not self.messages:
            raise ValueError("message or messages is required")
        return self


class ChatResponse(BaseModel):
    request_id: str
    model: str
    response: str
    routing_strategy: Literal["auto", "manual"]
    routing_reason: Optional[str] = None
    fallback_used: bool
    cache_type: Literal["exact", "semantic", "none"] = "none"