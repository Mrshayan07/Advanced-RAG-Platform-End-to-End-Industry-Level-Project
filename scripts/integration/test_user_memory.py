
import asyncio

from sqlalchemy import select

from app.db.models import Tenant, User
from app.db.session import AsyncSessionLocal
from app.services.user_memory import UserMemoryService


async def main() -> None:
    async with AsyncSessionLocal() as db:
        # ---------------------------------------------------------
        # 1. Find an existing user
        # ---------------------------------------------------------
        result = await db.execute(
            select(User).order_by(User.created_at.asc())
        )

        user = result.scalars().first()

        if user is None:
            print("ERROR: No users found in database.")
            return

        print("User found:")
        print(f"  user_id:   {user.id}")
        print(f"  tenant_id: {user.tenant_id}")
        print(f"  email:     {user.email}")

        # ---------------------------------------------------------
        # 2. Create service
        # ---------------------------------------------------------
        memory_service = UserMemoryService(db)

        # ---------------------------------------------------------
        # 3. Create memory
        # ---------------------------------------------------------
        memory = await memory_service.set_memory(
            tenant_id=user.tenant_id,
            user_id=user.id,
            memory_key="test_name",
            memory_value="Shayan Javed",
        )

        print("\nMemory created:")
        print(f"  key:   {memory.memory_key}")
        print(f"  value: {memory.memory_value}")

        # ---------------------------------------------------------
        # 4. Read memory
        # ---------------------------------------------------------
        fetched = await memory_service.get_memory(
            tenant_id=user.tenant_id,
            user_id=user.id,
            memory_key="test_name",
        )

        print("\nMemory fetched:")
        if fetched:
            print(f"  key:   {fetched.memory_key}")
            print(f"  value: {fetched.memory_value}")
        else:
            print("  ERROR: Memory not found.")

        # ---------------------------------------------------------
        # 5. Update same memory
        # ---------------------------------------------------------
        updated = await memory_service.set_memory(
            tenant_id=user.tenant_id,
            user_id=user.id,
            memory_key="test_name",
            memory_value="Shayan Javed Updated",
        )

        print("\nMemory updated:")
        print(f"  key:   {updated.memory_key}")
        print(f"  value: {updated.memory_value}")

        # ---------------------------------------------------------
        # 6. Get all memories
        # ---------------------------------------------------------
        memories = await memory_service.get_memories(
            tenant_id=user.tenant_id,
            user_id=user.id,
        )

        print("\nAll memories:")
        for item in memories:
            print(
                f"  {item.memory_key} = "
                f"{item.memory_value}"
            )

        # ---------------------------------------------------------
        # 7. Delete test memory
        # ---------------------------------------------------------
        deleted = await memory_service.delete_memory(
            tenant_id=user.tenant_id,
            user_id=user.id,
            memory_key="test_name",
        )

        print("\nDelete result:")
        print(f"  deleted: {deleted}")

        # ---------------------------------------------------------
        # 8. Verify deletion
        # ---------------------------------------------------------
        after_delete = await memory_service.get_memory(
            tenant_id=user.tenant_id,
            user_id=user.id,
            memory_key="test_name",
        )

        print("\nAfter deletion:")
        print(f"  memory: {after_delete}")


if __name__ == "__main__":
    asyncio.run(main())
