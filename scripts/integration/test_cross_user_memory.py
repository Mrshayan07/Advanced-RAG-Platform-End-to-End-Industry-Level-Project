import asyncio

from sqlalchemy import select

from app.db.models import Tenant, User
from app.db.session import AsyncSessionLocal
from app.services.user_memory import UserMemoryService


TEST_EMAIL_A = "memory-isolation-user-a@test.local"
TEST_EMAIL_B = "memory-isolation-user-b@test.local"
TEST_MEMORY_KEY = "isolation_test_name"


async def main() -> None:
    async with AsyncSessionLocal() as db:

        print("=" * 70)
        print("CROSS-USER / CROSS-TENANT MEMORY ISOLATION TEST")
        print("=" * 70)

        test_tenant_a = None
        test_tenant_b = None
        test_user_a = None
        test_user_b = None

        try:
            # =====================================================
            # CLEAN OLD TEST DATA IF IT EXISTS
            # =====================================================

            result = await db.execute(
                select(User).where(
                    User.email.in_(
                        [
                            TEST_EMAIL_A,
                            TEST_EMAIL_B,
                        ]
                    )
                )
            )

            old_users = list(result.scalars().all())

            for old_user in old_users:
                await db.delete(old_user)

            if old_users:
                await db.commit()

            # =====================================================
            # CREATE TEST TENANT A
            # =====================================================

            test_tenant_a = Tenant(
                name="Memory Isolation Test Tenant A"
            )

            db.add(test_tenant_a)
            await db.flush()

            # =====================================================
            # CREATE TEST USER A
            # =====================================================

            test_user_a = User(
                tenant_id=test_tenant_a.id,
                email=TEST_EMAIL_A,
                password_hash="TEST_ONLY_HASH",
                role="user",
            )

            db.add(test_user_a)

            # =====================================================
            # CREATE TEST TENANT B
            # =====================================================

            test_tenant_b = Tenant(
                name="Memory Isolation Test Tenant B"
            )

            db.add(test_tenant_b)
            await db.flush()

            # =====================================================
            # CREATE TEST USER B
            # =====================================================

            test_user_b = User(
                tenant_id=test_tenant_b.id,
                email=TEST_EMAIL_B,
                password_hash="TEST_ONLY_HASH",
                role="user",
            )

            db.add(test_user_b)

            await db.commit()

            print("\nTest users created:")

            print(
                f"\nUser A:"
                f"\n  email:     {test_user_a.email}"
                f"\n  user_id:   {test_user_a.id}"
                f"\n  tenant_id: {test_tenant_a.id}"
            )

            print(
                f"\nUser B:"
                f"\n  email:     {test_user_b.email}"
                f"\n  user_id:   {test_user_b.id}"
                f"\n  tenant_id: {test_tenant_b.id}"
            )

            # =====================================================
            # MEMORY SERVICE
            # =====================================================

            memory_service = UserMemoryService(db)

            # =====================================================
            # CREATE PRIVATE MEMORY FOR USER A
            # =====================================================

            await memory_service.set_memory(
                tenant_id=test_tenant_a.id,
                user_id=test_user_a.id,
                memory_key=TEST_MEMORY_KEY,
                memory_value="PRIVATE_USER_A_MEMORY",
                memory_type="profile",
            )

            print("\n" + "-" * 70)
            print("PRIVATE MEMORY CREATED")
            print("-" * 70)

            print("Owner: User A")
            print("Value: PRIVATE_USER_A_MEMORY")

            # =====================================================
            # TEST 1
            # USER A -> OWN MEMORY
            # =====================================================

            user_a_memory = await memory_service.get_memory(
                tenant_id=test_tenant_a.id,
                user_id=test_user_a.id,
                memory_key=TEST_MEMORY_KEY,
            )

            print("\n" + "-" * 70)
            print("TEST 1 — USER A -> OWN MEMORY")
            print("-" * 70)

            test_1 = user_a_memory is not None

            if test_1:
                print("PASS: User A can access its own memory.")
            else:
                print("FAIL: User A cannot access its own memory.")

            # =====================================================
            # TEST 2
            # USER B -> USER A MEMORY
            # =====================================================

            user_b_memory = await memory_service.get_memory(
                tenant_id=test_tenant_b.id,
                user_id=test_user_b.id,
                memory_key=TEST_MEMORY_KEY,
            )

            print("\n" + "-" * 70)
            print("TEST 2 — USER B -> USER A MEMORY")
            print("-" * 70)

            test_2 = user_b_memory is None

            if test_2:
                print(
                    "PASS: User B cannot access User A's memory."
                )
            else:
                print("FAIL: MEMORY LEAK DETECTED!")
                print(
                    f"Leaked value: "
                    f"{user_b_memory.memory_value}"
                )

            # =====================================================
            # TEST 3
            # USER A ID + USER B TENANT
            # =====================================================

            wrong_tenant_memory = await memory_service.get_memory(
                tenant_id=test_tenant_b.id,
                user_id=test_user_a.id,
                memory_key=TEST_MEMORY_KEY,
            )

            print("\n" + "-" * 70)
            print("TEST 3 — USER A ID + WRONG TENANT")
            print("-" * 70)

            test_3 = wrong_tenant_memory is None

            if test_3:
                print(
                    "PASS: User A memory is isolated "
                    "from another tenant."
                )
            else:
                print("FAIL: CROSS-TENANT MEMORY LEAK!")

            # =====================================================
            # TEST 4
            # USER B ID + USER A TENANT
            # =====================================================

            wrong_user_memory = await memory_service.get_memory(
                tenant_id=test_tenant_a.id,
                user_id=test_user_b.id,
                memory_key=TEST_MEMORY_KEY,
            )

            print("\n" + "-" * 70)
            print("TEST 4 — USER B ID + USER A TENANT")
            print("-" * 70)

            test_4 = wrong_user_memory is None

            if test_4:
                print(
                    "PASS: User B cannot access "
                    "User A's tenant memory."
                )
            else:
                print("FAIL: CROSS-USER MEMORY LEAK!")

            # =====================================================
            # TEST 5
            # LIST MEMORIES FOR USER B
            # =====================================================

            user_b_memories = await memory_service.get_memories(
                tenant_id=test_tenant_b.id,
                user_id=test_user_b.id,
            )

            print("\n" + "-" * 70)
            print("TEST 5 — USER B MEMORY LIST")
            print("-" * 70)

            test_5 = len(user_b_memories) == 0

            if test_5:
                print(
                    "PASS: User B memory list contains "
                    "no User A memory."
                )
            else:
                print("FAIL: User B received foreign memories!")

                for memory in user_b_memories:
                    print(
                        f"  {memory.memory_key} = "
                        f"{memory.memory_value}"
                    )

            # =====================================================
            # FINAL RESULT
            # =====================================================

            all_passed = (
                test_1
                and test_2
                and test_3
                and test_4
                and test_5
            )

            print("\n" + "=" * 70)

            if all_passed:
                print("ALL ISOLATION TESTS PASSED")
            else:
                print("ISOLATION TEST FAILED")

            print("=" * 70)

        finally:
            # =====================================================
            # CLEANUP
            # =====================================================

            print("\n" + "-" * 70)
            print("CLEANUP")
            print("-" * 70)

            if test_user_a is not None:
                result = await db.execute(
                    select(User).where(
                        User.id == test_user_a.id
                    )
                )

                user = result.scalar_one_or_none()

                if user is not None:
                    await db.delete(user)

            if test_user_b is not None:
                result = await db.execute(
                    select(User).where(
                        User.id == test_user_b.id
                    )
                )

                user = result.scalar_one_or_none()

                if user is not None:
                    await db.delete(user)

            await db.commit()

            if test_tenant_a is not None:
                result = await db.execute(
                    select(Tenant).where(
                        Tenant.id == test_tenant_a.id
                    )
                )

                tenant = result.scalar_one_or_none()

                if tenant is not None:
                    await db.delete(tenant)

            if test_tenant_b is not None:
                result = await db.execute(
                    select(Tenant).where(
                        Tenant.id == test_tenant_b.id
                    )
                )

                tenant = result.scalar_one_or_none()

                if tenant is not None:
                    await db.delete(tenant)

            await db.commit()

            print("Temporary test users removed.")
            print("Temporary test tenants removed.")
            print("Existing application data was not modified.")


if __name__ == "__main__":
    asyncio.run(main())