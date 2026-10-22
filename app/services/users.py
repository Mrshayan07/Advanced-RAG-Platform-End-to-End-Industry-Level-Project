from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.db.models import User


async def get_users_by_tenant(
    db: AsyncSession,
    tenant_id: UUID,
):
    result = await db.execute(
        select(User)
        .where(User.tenant_id == tenant_id)
        .order_by(User.created_at.asc())
    )

    return result.scalars().all()


async def get_user_by_id(
    db: AsyncSession,
    user_id: UUID,
    tenant_id: UUID,
):
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.tenant_id == tenant_id,
        )
    )

    return result.scalar_one_or_none()


async def create_tenant_user(
    db: AsyncSession,
    email: str,
    password: str,
    tenant_id: UUID,
):
    user = User(
        email=email,
        password_hash=hash_password(password),
        tenant_id=tenant_id,
        role="user",
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user