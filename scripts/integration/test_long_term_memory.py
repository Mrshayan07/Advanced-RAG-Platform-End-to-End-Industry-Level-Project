
"""
Long-Term Memory Service Integration Test

Tests:
1. Create Tenant + Users test fixtures
2. Create memory
3. Read memory
4. Update memory
5. Multiple memories
6. Memory type filtering
7. User isolation
8. Tenant isolation
9. Delete one memory
10. Clear all memories
11. Verify scoped cleanup
12. Cleanup temporary database fixtures

This test intentionally uses UserMemoryService directly instead of
MemoryExtractor so that persistence/isolation testing does not depend
on Gemini API availability or model behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from sqlalchemy import delete

from app.db.models import Tenant, User, UserMemory
from app.db.session import AsyncSessionLocal
from app.services.user_memory import UserMemoryService


async def main() -> None:
    print("=" * 70)
    print("LONG-TERM MEMORY SERVICE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Temporary test identities
    # ------------------------------------------------------------------

    tenant_a_id = uuid4()
    tenant_b_id = uuid4()

    user_a_id = uuid4()
    user_b_id = uuid4()

    user_a_email = f"ltm-test-a-{user_a_id}@example.com"
    user_b_email = f"ltm-test-b-{user_b_id}@example.com"

    async with AsyncSessionLocal() as db:
        service = UserMemoryService(db)

        try:
            # ==========================================================
            # 0. CREATE TEMPORARY TENANTS
            # ==========================================================

            print("\n[SETUP] Creating temporary tenants...")

            tenant_a = Tenant(
                id=tenant_a_id,
                name=f"LTM Test Tenant A {tenant_a_id}",
            )

            tenant_b = Tenant(
                id=tenant_b_id,
                name=f"LTM Test Tenant B {tenant_b_id}",
            )

            db.add_all([tenant_a, tenant_b])
            await db.flush()

            print("[PASS] Temporary tenants created.")

            # ==========================================================
            # 0.1 CREATE TEMPORARY USERS
            # ==========================================================

            print("\n[SETUP] Creating temporary users...")

            user_a = User(
                id=user_a_id,
                tenant_id=tenant_a_id,
                email=user_a_email,
                password_hash="test-password-hash",
                role="user",
            )

            user_b = User(
                id=user_b_id,
                tenant_id=tenant_a_id,
                email=user_b_email,
                password_hash="test-password-hash",
                role="user",
            )

            db.add_all([user_a, user_b])
            await db.commit()

            print("[PASS] Temporary users created.")

            # ==========================================================
            # 1. CREATE MEMORY
            # ==========================================================

            print("\n[TEST 1] Creating Tenant A / User A memory...")

            memory = await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
                memory_value="Shayan Javed",
                memory_type="profile",
            )

            assert memory.id is not None
            assert memory.tenant_id == tenant_a_id
            assert memory.user_id == user_a_id
            assert memory.memory_key == "name"
            assert memory.memory_value == "Shayan Javed"
            assert memory.memory_type == "profile"

            print("[PASS] Memory created successfully.")

            # ==========================================================
            # 2. READ MEMORY
            # ==========================================================

            print("\n[TEST 2] Reading memory...")

            fetched = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert fetched is not None
            assert fetched.memory_value == "Shayan Javed"
            assert fetched.tenant_id == tenant_a_id
            assert fetched.user_id == user_a_id

            print("[PASS] Memory retrieved successfully.")

            # ==========================================================
            # 3. UPDATE MEMORY
            # ==========================================================

            print("\n[TEST 3] Updating existing memory...")

            updated = await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
                memory_value="Shayan Ahmed",
                memory_type="profile",
            )

            assert updated.id == memory.id
            assert updated.memory_value == "Shayan Ahmed"

            fetched_after_update = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert fetched_after_update is not None
            assert fetched_after_update.memory_value == "Shayan Ahmed"

            print("[PASS] Existing memory updated instead of duplicated.")

            # ==========================================================
            # 4. CREATE MULTIPLE MEMORIES
            # ==========================================================

            print("\n[TEST 4] Creating multiple memories...")

            await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="university",
                memory_value="Sindh Madressatul Islam University",
                memory_type="education",
            )

            await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="degree",
                memory_value="BS Computer Science",
                memory_type="education",
            )

            await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="skill",
                memory_value="Python",
                memory_type="skill",
            )

            memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            assert len(memories) == 4

            memory_keys = {
                item.memory_key
                for item in memories
            }

            assert memory_keys == {
                "name",
                "university",
                "degree",
                "skill",
            }

            print("[PASS] Multiple memories stored correctly.")

            # ==========================================================
            # 5. MEMORY TYPE FILTER
            # ==========================================================

            print("\n[TEST 5] Testing memory_type filtering...")

            education_memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_type="education",
            )

            assert len(education_memories) == 2

            education_keys = {
                item.memory_key
                for item in education_memories
            }

            assert education_keys == {
                "university",
                "degree",
            }

            skill_memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_type="skill",
            )

            assert len(skill_memories) == 1
            assert skill_memories[0].memory_key == "skill"

            profile_memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_type="profile",
            )

            assert len(profile_memories) == 1
            assert profile_memories[0].memory_key == "name"

            print("[PASS] Memory type filtering works correctly.")

            # ==========================================================
            # 6. USER ISOLATION
            # ==========================================================

            print("\n[TEST 6] Testing user isolation...")

            # User B belongs to the same tenant but has separate memory.
            user_b_memory = await service.set_memory(
                tenant_id=tenant_a_id,
                user_id=user_b_id,
                memory_key="name",
                memory_value="User B",
                memory_type="profile",
            )

            assert user_b_memory.user_id == user_b_id
            assert user_b_memory.tenant_id == tenant_a_id

            # User A must only see User A memories.
            user_a_memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            assert len(user_a_memories) == 4

            for item in user_a_memories:
                assert item.user_id == user_a_id
                assert item.tenant_id == tenant_a_id

            # User A's "name" must remain its own value.
            user_a_name = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert user_a_name is not None
            assert user_a_name.memory_value == "Shayan Ahmed"

            # User B must see only User B's memory.
            user_b_memories = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_b_id,
            )

            assert len(user_b_memories) == 1
            assert user_b_memories[0].memory_key == "name"
            assert user_b_memories[0].memory_value == "User B"
            assert user_b_memories[0].user_id == user_b_id

            print("[PASS] User-level isolation verified.")

            # ==========================================================
            # 7. TENANT ISOLATION
            # ==========================================================

            print("\n[TEST 7] Testing tenant isolation...")

            # Same user UUID, different tenant.
            #
            # This is intentional: it verifies that tenant_id is part
            # of the memory isolation boundary.
            tenant_b_memory = await service.set_memory(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
                memory_key="name",
                memory_value="Tenant B User",
                memory_type="profile",
            )

            assert tenant_b_memory.tenant_id == tenant_b_id
            assert tenant_b_memory.user_id == user_a_id

            # Tenant A must return Tenant A memory.
            tenant_a_memory = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert tenant_a_memory is not None
            assert tenant_a_memory.tenant_id == tenant_a_id
            assert tenant_a_memory.memory_value == "Shayan Ahmed"

            # Tenant B must return Tenant B memory.
            tenant_b_fetched = await service.get_memory(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert tenant_b_fetched is not None
            assert tenant_b_fetched.tenant_id == tenant_b_id
            assert tenant_b_fetched.memory_value == "Tenant B User"

            # Tenant B must NOT see Tenant A's university memory.
            wrong_tenant_lookup = await service.get_memory(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
                memory_key="university",
            )

            assert wrong_tenant_lookup is None

            # Tenant B must only contain its own memory.
            tenant_b_memories = await service.get_memories(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
            )

            assert len(tenant_b_memories) == 1
            assert tenant_b_memories[0].memory_value == "Tenant B User"

            print("[PASS] Tenant-level isolation verified.")

            # ==========================================================
            # 8. DELETE ONE MEMORY
            # ==========================================================

            print("\n[TEST 8] Testing single-memory deletion...")

            deleted = await service.delete_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="skill",
            )

            assert deleted is True

            deleted_check = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="skill",
            )

            assert deleted_check is None

            # Deleting it again must return False.
            deleted_again = await service.delete_memory(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
                memory_key="skill",
            )

            assert deleted_again is False

            print("[PASS] Single-memory deletion works correctly.")

            # ==========================================================
            # 9. CLEAR ALL MEMORIES FOR USER A
            # ==========================================================

            print("\n[TEST 9] Testing clear_memories()...")

            remaining_before_clear = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            # name + university + degree = 3
            assert len(remaining_before_clear) == 3

            cleared_count = await service.clear_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            assert cleared_count == 3

            remaining_after_clear = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            assert remaining_after_clear == []

            print("[PASS] User memories cleared successfully.")

            # ==========================================================
            # 10. VERIFY CLEAR WAS PROPERLY SCOPED
            # ==========================================================

            print(
                "\n[TEST 10] Verifying clear_memories() "
                "does not affect other scopes..."
            )

            # User B belongs to same tenant.
            user_b_after_clear = await service.get_memory(
                tenant_id=tenant_a_id,
                user_id=user_b_id,
                memory_key="name",
            )

            assert user_b_after_clear is not None
            assert user_b_after_clear.memory_value == "User B"

            # Tenant B belongs to another tenant.
            tenant_b_after_clear = await service.get_memory(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
                memory_key="name",
            )

            assert tenant_b_after_clear is not None
            assert tenant_b_after_clear.memory_value == "Tenant B User"

            print(
                "[PASS] clear_memories() does not affect "
                "other users or tenants."
            )

            # ==========================================================
            # 11. VERIFY FINAL DATABASE STATE
            # ==========================================================

            print("\n[TEST 11] Verifying final memory state...")

            user_a_final = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_a_id,
            )

            user_b_final = await service.get_memories(
                tenant_id=tenant_a_id,
                user_id=user_b_id,
            )

            tenant_b_final = await service.get_memories(
                tenant_id=tenant_b_id,
                user_id=user_a_id,
            )

            assert user_a_final == []
            assert len(user_b_final) == 1
            assert len(tenant_b_final) == 1

            print("[PASS] Final scoped memory state is correct.")

            # ==========================================================
            # SUCCESS
            # ==========================================================

            print("\n" + "=" * 70)
            print("ALL LONG-TERM MEMORY TESTS PASSED")
            print("=" * 70)

        finally:
            # ==========================================================
            # CLEANUP
            # ==========================================================
            #
            # Important:
            #
            # 1. Delete user_memories first.
            # 2. Delete users second.
            # 3. Delete tenants last.
            #
            # This respects the foreign-key dependency chain:
            #
            # tenants
            #    ↓
            # users
            #    ↓
            # user_memories
            #
            # We use direct SQL deletes so cleanup does not depend on
            # ORM relationship loading.
            # ==========================================================

            print("\n[CLEANUP] Removing temporary test data...")

            try:
                # Delete memories belonging to both test tenants.
                await db.execute(
                    delete(UserMemory).where(
                        UserMemory.tenant_id.in_(
                            [tenant_a_id, tenant_b_id]
                        )
                    )
                )

                await db.flush()

                # Delete the temporary users.
                await db.execute(
                    delete(User).where(
                        User.id.in_(
                            [user_a_id, user_b_id]
                        )
                    )
                )

                await db.flush()

                # Delete the temporary tenants.
                await db.execute(
                    delete(Tenant).where(
                        Tenant.id.in_(
                            [tenant_a_id, tenant_b_id]
                        )
                    )
                )

                await db.commit()

                print("[CLEANUP] Temporary users, tenants, and memories removed.")

            except Exception:
                await db.rollback()
                print(
                    "[CLEANUP] Cleanup failed; database transaction "
                    "was rolled back."
                )
                raise


if __name__ == "__main__":
    asyncio.run(main())
