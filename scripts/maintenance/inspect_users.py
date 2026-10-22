import asyncio

from sqlalchemy import select

from app.db.models import User
from app.db.session import AsyncSessionLocal


async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User))

        users = result.scalars().all()

        for user in users:
            print(
                f"user_id={user.id} | "
                f"email={user.email} | "
                f"tenant_id={user.tenant_id} | "
                f"role={user.role}"
            )


if __name__ == "__main__":
    asyncio.run(main())
