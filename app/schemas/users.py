from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserListResponse(BaseModel):
    id: UUID
    email: EmailStr
    tenant_id: UUID
    role: str