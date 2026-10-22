from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: UUID
    name: str
    mime_type: str
    status: str
    tenant_id: UUID
    created_at: datetime