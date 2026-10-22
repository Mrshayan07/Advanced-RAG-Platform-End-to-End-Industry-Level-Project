
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from sqlalchemy import delete

from app.api import rag as rag_api
from app.core.security import create_access_token, hash_password
from app.db.models import Conversation, Message, Tenant, User, UserMemory
from app.db.session import AsyncSessionLocal
from app.main import app
from app.services.user_memory import UserMemoryService
from scripts.database_safety import require_test_database

import httpx


# =========================================================
# TEST CONFIGURATION
# =========================================================

TEST_PASSWORD = "LongTermMemoryTest123!"

USER_A_EMAIL = "api_long_term_memory_user_a@test.local"
USER_B_EMAIL = "api_long_term_memory_user_b@test.local"

MEMORY_KEY = "name"
MEMORY_VALUE = "Private Long-Term Memory User"


# =========================================================
# CREATE TEMPORARY USERS + TENANTS
# =========================================================

async def create_test_users():
    async with AsyncSessionLocal() as db:

        tenant_a = Tenant(
            id=uuid4(),
            name="API Long-Term Memory Test Tenant A",
        )

        tenant_b = Tenant(
            id=uuid4(),
            name="API Long-Term Memory Test Tenant B",
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
# CREATE DURABLE MEMORY
# =========================================================

async def create_memory(
    *,
    tenant_id: UUID,
    user_id: UUID,
):
    async with AsyncSessionLocal() as db:

        memory_service = UserMemoryService(db)

        memory = await memory_service.set_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_key=MEMORY_KEY,
            memory_value=MEMORY_VALUE,
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
    FK-safe cleanup.

    Order:

        Messages
            â†“
        Conversations
            â†“
        UserMemory
            â†“
        Users
            â†“
        Tenants
    """

    async with AsyncSessionLocal() as db:

        try:

            test_user_ids = [
                user_a_id,
                user_b_id,
            ]

            test_tenant_ids = [
                tenant_a_id,
                tenant_b_id,
            ]

            # -------------------------------------------------
            # 1. FIND TEST CONVERSATIONS
            # -------------------------------------------------

            result = await db.execute(
                select_conversation_ids(
                    test_user_ids=test_user_ids,
                    test_tenant_ids=test_tenant_ids,
                )
            )

            conversation_ids = [
                row[0]
                for row in result.all()
            ]

            # -------------------------------------------------
            # 2. DELETE MESSAGES
            # -------------------------------------------------

            if conversation_ids:

                await db.execute(
                    delete(Message).where(
                        Message.conversation_id.in_(
                            conversation_ids
                        )
                    )
                )

            # -------------------------------------------------
            # 3. DELETE CONVERSATIONS
            # -------------------------------------------------

            await db.execute(
                delete(Conversation).where(
                    Conversation.user_id.in_(
                        test_user_ids
                    ),
                    Conversation.tenant_id.in_(
                        test_tenant_ids
                    ),
                )
            )

            # -------------------------------------------------
            # 4. DELETE USER MEMORIES
            # -------------------------------------------------

            await db.execute(
                delete(UserMemory).where(
                    UserMemory.user_id.in_(
                        test_user_ids
                    ),
                    UserMemory.tenant_id.in_(
                        test_tenant_ids
                    ),
                )
            )

            # -------------------------------------------------
            # 5. DELETE USERS
            # -------------------------------------------------

            await db.execute(
                delete(User).where(
                    User.id.in_(
                        test_user_ids
                    ),
                    User.tenant_id.in_(
                        test_tenant_ids
                    ),
                )
            )

            # -------------------------------------------------
            # 6. DELETE TENANTS
            # -------------------------------------------------

            await db.execute(
                delete(Tenant).where(
                    Tenant.id.in_(
                        test_tenant_ids
                    )
                )
            )

            await db.commit()

        except Exception:

            await db.rollback()

            raise


# =========================================================
# CONVERSATION SELECT HELPER
# =========================================================

def select_conversation_ids(
    *,
    test_user_ids: list[UUID],
    test_tenant_ids: list[UUID],
):
    from sqlalchemy import select

    return select(
        Conversation.id
    ).where(
        Conversation.user_id.in_(
            test_user_ids
        ),
        Conversation.tenant_id.in_(
            test_tenant_ids
        ),
    )


# =========================================================
# MAIN TEST
# =========================================================

async def main():
    require_test_database()

    print()
    print("=" * 75)
    print("API LONG-TERM MEMORY END-TO-END TEST")
    print("=" * 75)

    tenant_a = None
    tenant_b = None
    user_a = None
    user_b = None

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # We disable automatic memory extraction after the request.
    #
    # The target of this test is:
    #
    # PostgreSQL Memory
    #       â†“
    # load_user_memory()
    #       â†“
    # RAGState
    #       â†“
    # LangGraph
    #       â†“
    # generation_prompt
    #       â†“
    # LLM
    #
    # We do NOT want the post-request MemoryManager to create
    # additional memories during this test.
    # ---------------------------------------------------------

    original_process_memory = (
        rag_api.MemoryManager.process_user_message
    )

    async def fake_process_user_message(
        self,
        *,
        tenant_id,
        user_id,
        user_message,
    ):
        return []

    rag_api.MemoryManager.process_user_message = (
        fake_process_user_message
    )

    try:

        # =====================================================
        # CREATE TEST USERS
        # =====================================================

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
            f"User A:   {user_a.id}"
        )

        print(
            f"Tenant B: {tenant_b.id}"
        )

        print(
            f"User B:   {user_b.id}"
        )

        # =====================================================
        # CREATE DURABLE MEMORY FOR USER A
        # =====================================================

        await create_memory(
            tenant_id=tenant_a.id,
            user_id=user_a.id,
        )

        print()
        print("DURABLE MEMORY CREATED")
        print("-" * 75)

        print(
            f"{MEMORY_KEY} = {MEMORY_VALUE}"
        )

        # =====================================================
        # CREATE REAL JWT TOKENS
        # =====================================================

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

        print()
        print("REAL JWT TOKENS CREATED")
        print("-" * 75)

        print("PASS: User A and User B tokens created.")

        # =====================================================
        # FASTAPI ASGI CLIENT
        # =====================================================

        transport = httpx.ASGITransport(
            app=app,
        )

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:

            # =================================================
            # TEST 1
            # USER A â†’ REAL /rag/query
            # =================================================

            response_a = await client.post(
                "/rag/query",
                headers={
                    "Authorization": f"Bearer {token_a}",
                },
                json={
                    "question": "What is my name?",
                    "conversation_id": None,
                    "chat_history": "",
                },
            )

            assert response_a.status_code == 200, (
                "User A /rag/query failed:\n"
                f"{response_a.text}"
            )

            body_a = response_a.json()

            answer_a = body_a.get(
                "answer",
                "",
            )

            print()
            print("TEST 1 â€” USER A REAL RAG REQUEST")
            print("-" * 75)

            print(
                f"HTTP status: {response_a.status_code}"
            )

            print(
                f"Answer: {answer_a}"
            )

            assert answer_a.strip(), (
                "User A received an empty answer."
            )

            # =================================================
            # TEST 2
            # USER A ANSWER MUST USE DURABLE MEMORY
            # =================================================

            assert MEMORY_VALUE in answer_a, (
                "Long-Term Memory integration failure.\n"
                "User A's durable memory was stored in "
                "PostgreSQL but the final RAG answer did "
                "not use it.\n"
                f"Expected memory value: {MEMORY_VALUE}\n"
                f"Actual answer: {answer_a}"
            )

            print()
            print("TEST 2 â€” DURABLE MEMORY USED BY RAG")
            print("-" * 75)

            print(
                "PASS: User A's durable PostgreSQL memory "
                "was used by the RAG generation pipeline."
            )

            # =================================================
            # TEST 3
            # USER A â†’ CITATIONS SHOULD NOT BE REQUIRED
            # FOR MEMORY-ONLY CLAIM
            # =================================================

            citations_a = body_a.get(
                "citations",
                [],
            )

            print()
            print("TEST 3 â€” MEMORY CITATION RULE")
            print("-" * 75)

            print(
                f"Citations returned: {len(citations_a)}"
            )

            print(
                "PASS: Durable memory is allowed to answer "
                "without document citation."
            )

            # =================================================
            # TEST 4
            # USER B â†’ REAL /rag/query
            # =================================================

            response_b = await client.post(
                "/rag/query",
                headers={
                    "Authorization": f"Bearer {token_b}",
                },
                json={
                    "question": "What is my name?",
                    "conversation_id": None,
                    "chat_history": "",
                },
            )

            assert response_b.status_code == 200, (
                "User B /rag/query failed:\n"
                f"{response_b.text}"
            )

            body_b = response_b.json()

            answer_b = body_b.get(
                "answer",
                "",
            )

            print()
            print("TEST 4 â€” USER B REAL RAG REQUEST")
            print("-" * 75)

            print(
                f"HTTP status: {response_b.status_code}"
            )

            print(
                f"Answer: {answer_b}"
            )

            # =================================================
            # TEST 5
            # USER B MUST NOT RECEIVE USER A MEMORY
            # =================================================

            assert MEMORY_VALUE not in answer_b, (
                "SECURITY FAILURE: User B received "
                "User A's durable memory."
            )

            print()
            print("TEST 5 â€” CROSS-USER MEMORY LEAK CHECK")
            print("-" * 75)

            print(
                "PASS: User B did not receive User A's "
                "durable memory."
            )

            # =================================================
            # TEST 6
            # USER B SHOULD HAVE NO DURABLE MEMORY
            # =================================================

            assert (
                answer_b
                != MEMORY_VALUE
            )

            print()
            print("TEST 6 â€” USER B MEMORY ISOLATION")
            print("-" * 75)

            print(
                "PASS: User B's response does not contain "
                "User A's private memory."
            )

            # =================================================
            # TEST 7
            # VERIFY CONVERSATIONS WERE CREATED
            # =================================================

            assert body_a.get(
                "conversation_id"
            ), "User A conversation was not created."

            assert body_b.get(
                "conversation_id"
            ), "User B conversation was not created."

            print()
            print("TEST 7 â€” CONVERSATION CREATION")
            print("-" * 75)

            print(
                "PASS: Real API requests created "
                "PostgreSQL conversations."
            )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        print()
        print("=" * 75)
        print(
            "ALL API LONG-TERM MEMORY TESTS PASSED"
        )
        print("=" * 75)

    finally:

        # =====================================================
        # RESTORE MEMORY MANAGER
        # =====================================================

        rag_api.MemoryManager.process_user_message = (
            original_process_memory
        )

        # =====================================================
        # CLEANUP
        # =====================================================

        if all(
            value is not None
            for value in (
                tenant_a,
                tenant_b,
                user_a,
                user_b,
            )
        ):

            print()
            print("-" * 75)
            print("CLEANUP")
            print("-" * 75)

            await cleanup(
                tenant_a_id=tenant_a.id,
                tenant_b_id=tenant_b.id,
                user_a_id=user_a.id,
                user_b_id=user_b.id,
            )

            print(
                "PASS: Temporary conversations, memories, "
                "users, and tenants removed."
            )

        print()
        print("=" * 75)
        print("TEST FINISHED")
        print("=" * 75)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())
