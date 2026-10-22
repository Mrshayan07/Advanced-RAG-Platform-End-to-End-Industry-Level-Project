
from pydantic import BaseModel, Field

from app.rag.citations.models import Citation


class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Conversation identifier for server-side short-term memory",
    )

    # Kept for backward compatibility.
    # Server-side Redis memory is preferred when conversation_id is provided.
    chat_history: str = Field(
        default="",
        description="Optional conversation history",
    )


class RAGQueryResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation]
