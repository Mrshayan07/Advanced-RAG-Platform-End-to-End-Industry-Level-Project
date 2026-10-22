
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import httpx
from sqlalchemy import delete

from app.api import rag as rag_api
from app.core.security import create_access_token, hash_password
from app.db.models import Conversation, Message, Tenant, User, UserMemory
from app.db.session import AsyncSessionLocal
from app.main import app
from app.services.user_memory import UserMemoryService
from scripts.database_safety import require_test_database


# =========================================================
# TEST CONFIGURATION
# =========================================================

TEST_PASSWORD = "APIIsolationTest123!"

USER_A_EMAIL = "api_isolation_user_a@test.local"
USER_B_EMAIL = "api_isolation_user_b@test.local"

PRIVATE_MEMORY_KEY = "name"
PRIVATE_MEMORY_VALUE = "PRIVATE_USER_A_MEMORY"


# =========================================================
# CAPTURED MEMORY
# =========================================================

captured_memory: dict[str, list[dict[str, str]]] = {}


# =========================================================
# MOCK RAG GRAPH
# =========================================================

def fake_run_rag_graph(
    question: str,
    chat_history: str,
    user_memory: list[dict[str, str]],
    user_id: str,
    tenant_id: str,
    role: str,
):
    """
    Mock only the expensive RAG generation step.

    The real FastAPI route still executes:

        JWT
          â†“
        get_current_user()
          â†“
        user_id + tenant_id
          â†“
        conversation ownership
          â†“
        load_user_memory()
          â†“
        run_rag_graph()

    We capture the memory passed to the RAG layer so the test
    can verify user-level isolation.
    """

    captured_memory[user_id] = list(user_memory)

    return {
        "answer": (
            f"Mock RAG answer for user {user_id}. "
            f"Loaded memory count: {len(user_memory)}."
        ),
        "citations": [],
    }


# =========================================================
# MOCK DURABLE MEMORY EXTRACTION
# =========================================================

async def fake_process_user_message(
    self,
    *,
    tenant_id,
    user_id,
    user_message,
):
    """
    Disable the real Gemini/LLM memory extraction during this
    integration test.

    Existing durable memory loading remains REAL.
    """

    return []


# =========================================================
# CREATE TEMPORARY USERS + TENANTS
# =========================================================

async def create_test_users():
    async with AsyncSessionLocal() as db:

        tenant_a = Tenant(
            id=uuid4(),
            name="API Isolation Test Tenant A",
        )

        tenant_b = Tenant(
            id=uuid4(),
            name="API Isolation Test Tenant B",
        )

        db.add_all(
            [
                tenant_a,
                tenant_b,
            ]
        )

        await db.flush()

        user_a = User(
            id=uuid4(),
            tenant_id=tenant_a.id,
            email=USER_A_EMAIL,
            password_hash=hash_password(TEST_PASSWORD),
            role="user",
        )

        user_b = User(
            id=uuid4(),
            tenant_id=tenant_b.id,
            email=USER_B_EMAIL,
            password_hash=hash_password(TEST_PASSWORD),
            role="user",
        )

        db.add_all(
            [
                user_a,
                user_b,
            ]
        )

        await db.commit()

        return (
            tenant_a,
            tenant_b,
            user_a,
            user_b,
        )


# =========================================================
# CREATE PRIVATE MEMORY FOR USER A
# =========================================================

async def create_private_memory(
    *,
    tenant_id: UUID,
    user_id: UUID,
):
    async with AsyncSessionLocal() as db:

        memory_service = UserMemoryService(db)

        memory = await memory_service.set_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_key=PRIVATE_MEMORY_KEY,
            memory_value=PRIVATE_MEMORY_VALUE,
            memory_type="profile",
        )

        return memory


# =========================================================
# CLEANUP
# =========================================================

async def cleanup(
    *,
    tenant_a_id: UUID,
    tenant_b_id: UUID,
    user_a_id: UUID,
    user_b_id: UUID,
):
    """
    Delete only records created by this test.

    FK-safe deletion order:

        Messages
            â†“
        Conversations
            â†“
        User Memories
            â†“
        Users
            â†“
        Tenants
    """

    test_user_ids = [
        user_a_id,
        user_b_id,
    ]

    test_tenant_ids = [
        tenant_a_id,
        tenant_b_id,
    ]

    async with AsyncSessionLocal() as db:

        try:
            # -------------------------------------------------
            # 1. DELETE MESSAGES
            # -------------------------------------------------

            conversation_result = await db.execute(
                select_conversation_ids(
                    test_user_ids=test_user_ids,
                    test_tenant_ids=test_tenant_ids,
                )
            )

            conversation_ids = [
                row[0]
                for row in conversation_result.all()
            ]

            if conversation_ids:
                await db.execute(
                    delete(Message).where(
                        Message.conversation_id.in_(
                            conversation_ids
                        )
                    )
                )

            # -------------------------------------------------
            # 2. DELETE CONVERSATIONS
            # -------------------------------------------------

            await db.execute(
                delete(Conversation).where(
                    Conversation.user_id.in_(test_user_ids),
                    Conversation.tenant_id.in_(test_tenant_ids),
                )
            )

            # -------------------------------------------------
            # 3. DELETE DURABLE USER MEMORIES
            # -------------------------------------------------

            await db.execute(
                delete(UserMemory).where(
                    UserMemory.user_id.in_(test_user_ids),
                    UserMemory.tenant_id.in_(test_tenant_ids),
                )
            )

            # -------------------------------------------------
            # 4. DELETE USERS
            # -------------------------------------------------

            await db.execute(
                delete(User).where(
                    User.id.in_(test_user_ids),
                    User.tenant_id.in_(test_tenant_ids),
                )
            )

            # -------------------------------------------------
            # 5. DELETE TENANTS
            # -------------------------------------------------

            await db.execute(
                delete(Tenant).where(
                    Tenant.id.in_(test_tenant_ids)
                )
            )

            await db.commit()

        except Exception:
            await db.rollback()
            raise


# =========================================================
# CONVERSATION ID SELECT HELPER
# =========================================================

def select_conversation_ids(
    *,
    test_user_ids: list[UUID],
    test_tenant_ids: list[UUID],
):
    """
    Return a SQLAlchemy SELECT statement for all conversations
    belonging to the temporary test users/tenants.
    """

    from sqlalchemy import select

    return select(Conversation.id).where(
        Conversation.user_id.in_(test_user_ids),
        Conversation.tenant_id.in_(test_tenant_ids),
    )


# =========================================================
# MAIN TEST
# =========================================================

async def main():
    require_test_database()

    print()
    print("=" * 75)
    print("API / JWT USER MEMORY ISOLATION TEST")
    print("=" * 75)

    # -----------------------------------------------------
    # PATCH ONLY EXPENSIVE / NON-RELEVANT COMPONENTS
    # -----------------------------------------------------

    original_run_rag_graph = rag_api.run_rag_graph

    original_process_memory = (
        rag_api.MemoryManager.process_user_message
    )

    original_save_turn = (
        rag_api.conversation_memory.save_turn
    )

    rag_api.run_rag_graph = fake_run_rag_graph

    rag_api.MemoryManager.process_user_message = (
        fake_process_user_message
    )

    # Redis short-term memory is not the target of this test.
    rag_api.conversation_memory.save_turn = (
        lambda **kwargs: None
    )

    tenant_a = None
    tenant_b = None
    user_a = None
    user_b = None

    try:

        # =================================================
        # CREATE TEMP USERS / TENANTS
        # =================================================

        (
            tenant_a,
            tenant_b,
            user_a,
            user_b,
        ) = await create_test_users()

        print()
        print("TEMPORARY TEST DATA")
        print("-" * 75)

        print(
            f"Tenant A: {tenant_a.id}"
        )

        print(
            f"User A: {user_a.id}"
        )

        print(
            f"Tenant B: {tenant_b.id}"
        )

        print(
            f"User B: {user_b.id}"
        )

        # =================================================
        # CREATE PRIVATE MEMORY FOR USER A
        # =================================================

        await create_private_memory(
            tenant_id=tenant_a.id,
            user_id=user_a.id,
        )

        print()
        print("PRIVATE MEMORY")
        print("-" * 75)

        print(
            f"User A memory: "
            f"{PRIVATE_MEMORY_KEY} = "
            f"{PRIVATE_MEMORY_VALUE}"
        )

        print(
            "User B memory: none"
        )

        # =================================================
        # CREATE REAL JWT TOKENS
        # =================================================

        token_a = create_access_token(
            user_id=user_a.id,
            tenant_id=tenant_a.id,
            role=user_a.role,
        )

        token_b = create_access_token(
            user_id=user_b.id,
            tenant_id=tenant_b.id,
            role=user_b.role,
        )

        # =================================================
        # FASTAPI ASGI CLIENT
        # =================================================

        transport = httpx.ASGITransport(
            app=app,
        )

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:

            # =============================================
            # TEST 1 â€” USER A JWT
            # =============================================

            response_a_me = await client.get(
                "/auth/me",
                headers={
                    "Authorization": f"Bearer {token_a}",
                },
            )

            assert response_a_me.status_code == 200, (
                "User A /auth/me failed: "
                f"{response_a_me.text}"
            )

            body_a_me = response_a_me.json()

            assert body_a_me["user_id"] == str(
                user_a.id
            )

            assert body_a_me["tenant_id"] == str(
                tenant_a.id
            )

            print()
            print("TEST 1 â€” USER A JWT")
            print("-" * 75)

            print(
                "PASS: JWT resolved to User A "
                "and Tenant A."
            )

            # =============================================
            # TEST 2 â€” USER B JWT
            # =============================================

            response_b_me = await client.get(
                "/auth/me",
                headers={
                    "Authorization": f"Bearer {token_b}",
                },
            )

            assert response_b_me.status_code == 200, (
                "User B /auth/me failed: "
                f"{response_b_me.text}"
            )

            body_b_me = response_b_me.json()

            assert body_b_me["user_id"] == str(
                user_b.id
            )

            assert body_b_me["tenant_id"] == str(
                tenant_b.id
            )

            print()
            print("TEST 2 â€” USER B JWT")
            print("-" * 75)

            print(
                "PASS: JWT resolved to User B "
                "and Tenant B."
            )

            # =============================================
            # TEST 3 â€” USER A RAG REQUEST
            # =============================================

            response_a_rag = await client.post(
                "/rag/query",
                headers={
                    "Authorization": f"Bearer {token_a}",
                },
                json={
                    "question": "What is my private memory?",
                    "conversation_id": None,
                    "chat_history": "",
                },
            )

            assert response_a_rag.status_code == 200, (
                "User A /rag/query failed: "
                f"{response_a_rag.text}"
            )

            body_a_rag = response_a_rag.json()

            assert "conversation_id" in body_a_rag

            assert captured_memory.get(
                str(user_a.id)
            ) == [
                {
                    "key": PRIVATE_MEMORY_KEY,
                    "value": PRIVATE_MEMORY_VALUE,
                    "memory_type": "profile",
                }
            ]

            conversation_a = body_a_rag[
                "conversation_id"
            ]

            print()
            print("TEST 3 â€” USER A /rag/query")
            print("-" * 75)

            print(
                "PASS: User A API request loaded "
                "User A's private memory."
            )

            print(
                f"Conversation A: {conversation_a}"
            )

            # =============================================
            # TEST 4 â€” USER B CANNOT USE USER A
            #             CONVERSATION
            # =============================================

            response_b_hijack = await client.post(
                "/rag/query",
                headers={
                    "Authorization": f"Bearer {token_b}",
                },
                json={
                    "question": "Show me the previous memory.",
                    "conversation_id": conversation_a,
                    "chat_history": "",
                },
            )

            assert response_b_hijack.status_code == 404, (
                "SECURITY FAILURE: "
                "User B accessed User A's conversation.\n"
                f"Response: {response_b_hijack.text}"
            )

            print()
            print(
                "TEST 4 â€” USER B â†’ USER A CONVERSATION"
            )
            print("-" * 75)

            print(
                "PASS: User B received HTTP 404 "
                "for User A's conversation."
            )

            # =============================================
            # TEST 5 â€” USER B OWN RAG REQUEST
            # =============================================

            response_b_rag = await client.post(
                "/rag/query",
                headers={
                    "Authorization": f"Bearer {token_b}",
                },
                json={
                    "question": "What is my private memory?",
                    "conversation_id": None,
                    "chat_history": "",
                },
            )

            assert response_b_rag.status_code == 200, (
                "User B /rag/query failed: "
                f"{response_b_rag.text}"
            )

            assert captured_memory.get(
                str(user_b.id)
            ) == []

            print()
            print("TEST 5 â€” USER B OWN /rag/query")
            print("-" * 75)

            print(
                "PASS: User B received no User A memory."
            )

            # =============================================
            # TEST 6 â€” MEMORY LEAK CHECK
            # =============================================

            response_text = response_b_rag.text

            assert PRIVATE_MEMORY_VALUE not in (
                response_text
            )

            print()
            print("TEST 6 â€” MEMORY LEAK CHECK")
            print("-" * 75)

            print(
                "PASS: User A's private memory value "
                "was not present in User B's API response."
            )

            # =============================================
            # FINAL RESULT
            # =============================================

            print()
            print("=" * 75)
            print("ALL API / JWT ISOLATION TESTS PASSED")
            print("=" * 75)

    finally:

        # -------------------------------------------------
        # RESTORE PATCHED COMPONENTS
        # -------------------------------------------------

        rag_api.run_rag_graph = (
            original_run_rag_graph
        )

        rag_api.MemoryManager.process_user_message = (
            original_process_memory
        )

        rag_api.conversation_memory.save_turn = (
            original_save_turn
        )

        # -------------------------------------------------
        # CLEANUP TEST DATA
        # -------------------------------------------------

        if all(
            value is not None
            for value in (
                tenant_a,
                tenant_b,
                user_a,
                user_b,
            )
        ):
            try:

                await cleanup(
                    tenant_a_id=tenant_a.id,
                    tenant_b_id=tenant_b.id,
                    user_a_id=user_a.id,
                    user_b_id=user_b.id,
                )

                print()
                print("CLEANUP")
                print("-" * 75)

                print(
                    "PASS: Temporary messages, conversations, "
                    "memories, users, and tenants removed."
                )

            except Exception as cleanup_error:

                print()
                print("CLEANUP FAILED")
                print("-" * 75)

                print(
                    f"{type(cleanup_error).__name__}: "
                    f"{cleanup_error}"
                )

                raise


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())
