from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    citations: list[dict] = Field(
        default_factory=list
    )

    model_config = ConfigDict(
        from_attributes=True,
    )


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class ConversationDetailResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    messages: list[MessageResponse]

    model_config = ConfigDict(
        from_attributes=True,
    )