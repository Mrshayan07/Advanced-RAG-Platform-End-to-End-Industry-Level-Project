from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_tenant_id,
    get_current_user,
    get_db,
    require_role,
)
from app.core.security import (
    create_access_token,
    verify_password,
)
from app.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth import (
    create_user,
    get_user_by_email,
)
from app.services.documents import get_documents_by_tenant

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    existing_user = await get_user_by_email(
        db,
        data.email,
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    user = await create_user(
        db=db,
        email=data.email,
        password=data.password,
        tenant_name=data.tenant_name,
    )

    return user


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    user = await get_user_by_email(
        db,
        data.email,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(
        data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get(
    "/me",
    response_model=CurrentUserResponse,
)
async def get_me(
    current_user=Depends(get_current_user),
):
    return current_user

@router.get("/admin-only")
async def admin_only(
    current_user=Depends(require_role("admin")),
):
    return {
        "message": "Admin access granted",
        "user_id": str(current_user["user_id"]),
        "role": current_user["role"],
    }

@router.get("/tenant")
async def get_my_tenant(
    tenant_id=Depends(get_current_tenant_id),
):
    return {
        "message": "Tenant access granted",
        "tenant_id": str(tenant_id),
    }

@router.get("/tenant-documents")
async def get_my_tenant_documents(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    documents = await get_documents_by_tenant(
        db=db,
        tenant_id=current_user["tenant_id"],
    )

    return {
        "tenant_id": str(current_user["tenant_id"]),
        "documents": [
            {
                "id": str(document.id),
                "name": document.name,
                "status": document.status,
            }
            for document in documents
        ],
    }