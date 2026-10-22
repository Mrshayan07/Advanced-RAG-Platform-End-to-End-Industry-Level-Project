import asyncio

from sqlalchemy import func, select

from app.db.models import User, UserMemory
from app.db.session import AsyncSessionLocal
from app.services.memory_manager import MemoryManager


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

        print("User:")
        print(f"  user_id:   {user.id}")
        print(f"  tenant_id: {user.tenant_id}")
        print(f"  email:     {user.email}")

        # ---------------------------------------------------------
        # 2. Create MemoryManager
        # ---------------------------------------------------------
        manager = MemoryManager(db)

        # ---------------------------------------------------------
        # 3. Clean previous test memories
        # ---------------------------------------------------------
        test_keys = [
            "name",
            "university",
            "degree",
            "job_title",
            "skill",
        ]

        existing_result = await db.execute(
            select(UserMemory).where(
                UserMemory.tenant_id == user.tenant_id,
                UserMemory.user_id == user.id,
                UserMemory.memory_key.in_(test_keys),
            )
        )

        existing_memories = list(
            existing_result.scalars().all()
        )

        for memory in existing_memories:
            await db.delete(memory)

        await db.commit()

        print("\nPrevious test memories cleared.")

        # ---------------------------------------------------------
        # 4. Test name extraction + persistence
        # ---------------------------------------------------------
        message = "My name is Shayan Javed."

        print("\n" + "=" * 70)
        print(f"USER: {message}")

        persisted = await manager.process_user_message(
            tenant_id=user.tenant_id,
            user_id=user.id,
            user_message=message,
        )

        print("PERSISTED:")

        for memory in persisted:
            print(
                f"  {memory['key']} = "
                f"{memory['value']} "
                f"[{memory['memory_type']}]"
            )

        # ---------------------------------------------------------
        # 5. Test education extraction + persistence
        # ---------------------------------------------------------
        message = (
            "I study BS Computer Science at "
            "Sindh Madressatul Islam University."
        )

        print("\n" + "=" * 70)
        print(f"USER: {message}")

        persisted = await manager.process_user_message(
            tenant_id=user.tenant_id,
            user_id=user.id,
            user_message=message,
        )

        print("PERSISTED:")

        for memory in persisted:
            print(
                f"  {memory['key']} = "
                f"{memory['value']} "
                f"[{memory['memory_type']}]"
            )

        # ---------------------------------------------------------
        # 6. Test question should create no memory
        # ---------------------------------------------------------
        message = "What is my name?"

        print("\n" + "=" * 70)
        print(f"USER: {message}")

        persisted = await manager.process_user_message(
            tenant_id=user.tenant_id,
            user_id=user.id,
            user_message=message,
        )

        print("PERSISTED:")

        if not persisted:
            print("  No memories saved.")
        else:
            for memory in persisted:
                print(
                    f"  {memory['key']} = "
                    f"{memory['value']} "
                    f"[{memory['memory_type']}]"
                )

        # ---------------------------------------------------------
        # 7. Test password should create no memory
        # ---------------------------------------------------------
        message = "My password is Test123!"

        print("\n" + "=" * 70)
        print(f"USER: {message}")

        persisted = await manager.process_user_message(
            tenant_id=user.tenant_id,
            user_id=user.id,
            user_message=message,
        )

        print("PERSISTED:")

        if not persisted:
            print("  No memories saved.")
        else:
            for memory in persisted:
                print(
                    f"  {memory['key']} = "
                    f"{memory['value']} "
                    f"[{memory['memory_type']}]"
                )

        # ---------------------------------------------------------
        # 8. Test memory update / overwrite
        # ---------------------------------------------------------
        message = "My name is Shayan Ahmed."

        print("\n" + "=" * 70)
        print("MEMORY UPDATE / OVERWRITE TEST")
        print("=" * 70)
        print(f"USER: {message}")

        persisted = await manager.process_user_message(
            tenant_id=user.tenant_id,
            user_id=user.id,
            user_message=message,
        )

        print("PERSISTED:")

        for memory in persisted:
            print(
                f"  {memory['key']} = "
                f"{memory['value']} "
                f"[{memory['memory_type']}]"
            )

        # ---------------------------------------------------------
        # 9. Verify name was updated, not duplicated
        # ---------------------------------------------------------
        name_result = await db.execute(
            select(UserMemory).where(
                UserMemory.tenant_id == user.tenant_id,
                UserMemory.user_id == user.id,
                UserMemory.memory_key == "name",
            )
        )

        name_memories = list(
            name_result.scalars().all()
        )

        print("\n" + "=" * 70)
        print("NAME UPDATE VERIFICATION")
        print("=" * 70)

        print(
            f"Name memory count: "
            f"{len(name_memories)}"
        )

        assert len(name_memories) == 1, (
            "FAIL: Duplicate name memories found."
        )

        updated_name = name_memories[0]

        print(
            f"Current name: "
            f"{updated_name.memory_value}"
        )

        assert (
            updated_name.memory_value
            == "Shayan Ahmed"
        ), (
            "FAIL: Name was not updated correctly."
        )

        assert (
            updated_name.memory_value
            != "Shayan Javed"
        ), (
            "FAIL: Old name still exists."
        )

        print(
            "PASS: Existing name memory was "
            "updated without creating a duplicate."
        )

        # ---------------------------------------------------------
        # 10. Verify all database memories
        # ---------------------------------------------------------
        result = await db.execute(
            select(UserMemory)
            .where(
                UserMemory.tenant_id == user.tenant_id,
                UserMemory.user_id == user.id,
            )
            .order_by(UserMemory.memory_key.asc())
        )

        memories = list(
            result.scalars().all()
        )

        print("\n" + "=" * 70)
        print("DATABASE MEMORIES:")
        print("=" * 70)

        for memory in memories:
            print(
                f"  {memory.memory_key} = "
                f"{memory.memory_value} "
                f"[{memory.memory_type}]"
            )

        # ---------------------------------------------------------
        # 11. Cleanup test memories
        # ---------------------------------------------------------
        for memory in memories:
            await db.delete(memory)

        await db.commit()

        print("\nTest memories cleaned up.")
        print("MemoryManager test completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())