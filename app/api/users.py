from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_role
from app.schemas.users import CreateUserRequest, UserListResponse
from app.services.auth import get_user_by_email
from app.services.users import (
    create_tenant_user,
    get_users_by_tenant,
)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.get(
    "",
    response_model=list[UserListResponse],
)
async def list_users(
    current_user=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    users = await get_users_by_tenant(
        db=db,
        tenant_id=current_user["tenant_id"],
    )

    return users


@router.post(
    "",
    response_model=UserListResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    data: CreateUserRequest,
    current_user=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    existing_user = await get_user_by_email(
        db=db,
        email=data.email,
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    user = await create_tenant_user(
        db=db,
        email=data.email,
        password=data.password,
        tenant_id=current_user["tenant_id"],
    )

    return user