from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.db.models import Tenant, User


async def get_user_by_email(
    db: AsyncSession,
    email: str,
):
    result = await db.execute(
        select(User).where(User.email == email)
    )

    return result.scalar_one_or_none()





async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    tenant_name: str,
):
    tenant = Tenant(name=tenant_name)

    db.add(tenant)
    await db.flush()

    user = User(
        email=email,
        password_hash=hash_password(password),
        tenant_id=tenant.id,
        role="admin",
    )

    db.add(user)

    await db.commit()
    await db.refresh(user)

    return user