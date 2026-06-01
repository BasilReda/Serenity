from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    user_id: str

class SSEEvent(BaseModel):
    type: str           # "status" | "response" | "error" | "done"
    node: Optional[str] = None
    data: str