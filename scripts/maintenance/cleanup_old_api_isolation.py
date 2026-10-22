import asyncio

from sqlalchemy import delete, select

from app.db.session import AsyncSessionLocal
from app.db.models import Conversation, Message, Tenant, User, UserMemory


TEST_EMAILS = [
    "api_isolation_user_a@test.local",
    "api_isolation_user_b@test.local",
]


async def cleanup_old_test_data():
    async with AsyncSessionLocal() as db:
        try:
            # -----------------------------------------------------
            # 1. Find old test users
            # -----------------------------------------------------
            result = await db.execute(
                select(User.id, User.tenant_id).where(
                    User.email.in_(TEST_EMAILS)
                )
            )

            rows = result.all()

            if not rows:
                print("No old API isolation test users found.")
                return

            user_ids = [row[0] for row in rows]
            tenant_ids = list({row[1] for row in rows})

            print(f"Found {len(user_ids)} old test user(s).")

            # -----------------------------------------------------
            # 2. Find conversations belonging to those users
            # -----------------------------------------------------
            result = await db.execute(
                select(Conversation.id).where(
                    Conversation.user_id.in_(user_ids)
                )
            )

            conversation_ids = list(result.scalars().all())

            print(
                f"Found {len(conversation_ids)} old test conversation(s)."
            )

            # -----------------------------------------------------
            # 3. Delete messages FIRST
            # -----------------------------------------------------
            if conversation_ids:
                await db.execute(
                    delete(Message).where(
                        Message.conversation_id.in_(conversation_ids)
                    )
                )
                print("Deleted old test messages.")

            # -----------------------------------------------------
            # 4. Delete conversations
            # -----------------------------------------------------
            if conversation_ids:
                await db.execute(
                    delete(Conversation).where(
                        Conversation.id.in_(conversation_ids)
                    )
                )
                print("Deleted old test conversations.")

            # -----------------------------------------------------
            # 5. Delete durable memories
            # -----------------------------------------------------
            await db.execute(
                delete(UserMemory).where(
                    UserMemory.user_id.in_(user_ids)
                )
            )
            print("Deleted old test user memories.")

            # -----------------------------------------------------
            # 6. Delete users
            # -----------------------------------------------------
            await db.execute(
                delete(User).where(
                    User.id.in_(user_ids)
                )
            )
            print("Deleted old test users.")

            # -----------------------------------------------------
            # 7. Delete tenants ONLY if no other users remain
            # -----------------------------------------------------
            for tenant_id in tenant_ids:
                remaining_users = await db.execute(
                    select(User.id).where(
                        User.tenant_id == tenant_id
                    )
                )

                if remaining_users.first() is None:
                    await db.execute(
                        delete(Tenant).where(
                            Tenant.id == tenant_id
                        )
                    )
                    print(f"Deleted old test tenant: {tenant_id}")
                else:
                    print(
                        f"Kept tenant {tenant_id}: "
                        "other users still belong to it."
                    )

            await db.commit()

            print()
            print("=" * 70)
            print("OLD API ISOLATION TEST DATA CLEANUP PASSED")
            print("=" * 70)

        except Exception:
            await db.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(cleanup_old_test_data())
