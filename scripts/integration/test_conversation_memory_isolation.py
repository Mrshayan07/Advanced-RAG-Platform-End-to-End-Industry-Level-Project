from __future__ import annotations

"""
Short-Term Conversation Memory Isolation Test

Tests:
1. Redis connectivity
2. Conversation-level isolation
3. History construction
4. Summary isolation
5. Message limit / compression
6. Conversation clear behavior

This test intentionally does NOT import the FastAPI RAG API.
That prevents loading the expensive RAG graph and cross-encoder model.
"""

import uuid

from app.services.memory import ConversationMemory


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------------------------

def main() -> None:
    print_section("SHORT-TERM CONVERSATION MEMORY ISOLATION TEST")

    memory = ConversationMemory(
        message_limit=10,
    )

    # -----------------------------------------------------------------------
    # TEST 1 — REDIS CONNECTIVITY
    # -----------------------------------------------------------------------

    print_section("TEST 1 — REDIS CONNECTIVITY")

    redis_ok = memory.ping()

    assert_true(
        redis_ok,
        "Redis ping failed.",
    )

    print("PASS: Redis connection is healthy.")

    # -----------------------------------------------------------------------
    # CREATE UNIQUE CONVERSATIONS
    # -----------------------------------------------------------------------

    conversation_a = memory.create_conversation_id()
    conversation_b = memory.create_conversation_id()

    print()
    print("Temporary conversations:")
    print("-" * 75)
    print(f"Conversation A: {conversation_a}")
    print(f"Conversation B: {conversation_b}")

    # Always start clean.
    memory.clear(conversation_a)
    memory.clear(conversation_b)

    try:
        # -------------------------------------------------------------------
        # TEST 2 — CONVERSATION A MEMORY
        # -------------------------------------------------------------------

        print_section("TEST 2 — CONVERSATION A MEMORY")

        memory.save_turn(
            conversation_id=conversation_a,
            question="What is my project?",
            answer="You are building an Advanced RAG Platform.",
        )

        history_a = memory.build_chat_history(
            conversation_id=conversation_a,
        )

        assert_true(
            "What is my project?" in history_a,
            "Conversation A question was not found.",
        )

        assert_true(
            "Advanced RAG Platform" in history_a,
            "Conversation A answer was not found.",
        )

        print("PASS: Conversation A history stored correctly.")

        # -------------------------------------------------------------------
        # TEST 3 — CONVERSATION B MUST NOT SEE A
        # -------------------------------------------------------------------

        print_section("TEST 3 — CONVERSATION ISOLATION")

        history_b_before = memory.build_chat_history(
            conversation_id=conversation_b,
        )

        assert_true(
            history_b_before == "",
            "Conversation B unexpectedly contains data before writing.",
        )

        assert_true(
            "What is my project?" not in history_b_before,
            "Conversation B can see Conversation A question.",
        )

        assert_true(
            "Advanced RAG Platform" not in history_b_before,
            "Conversation B can see Conversation A answer.",
        )

        print(
            "PASS: Conversation B cannot access Conversation A history."
        )

        # -------------------------------------------------------------------
        # TEST 4 — B GETS ITS OWN MEMORY
        # -------------------------------------------------------------------

        print_section("TEST 4 — CONVERSATION B OWN MEMORY")

        memory.save_turn(
            conversation_id=conversation_b,
            question="What database am I using?",
            answer="The project uses PostgreSQL.",
        )

        history_b = memory.build_chat_history(
            conversation_id=conversation_b,
        )

        assert_true(
            "What database am I using?" in history_b,
            "Conversation B question was not stored.",
        )

        assert_true(
            "PostgreSQL" in history_b,
            "Conversation B answer was not stored.",
        )

        assert_true(
            "What is my project?" not in history_b,
            "Conversation B contains Conversation A question.",
        )

        assert_true(
            "Advanced RAG Platform" not in history_b,
            "Conversation B contains Conversation A answer.",
        )

        print(
            "PASS: Conversation B contains only its own conversation."
        )

        # -------------------------------------------------------------------
        # TEST 5 — A STILL CANNOT SEE B
        # -------------------------------------------------------------------

        print_section("TEST 5 — BIDIRECTIONAL ISOLATION")

        history_a_after_b = memory.build_chat_history(
            conversation_id=conversation_a,
        )

        assert_true(
            "What is my project?" in history_a_after_b,
            "Conversation A original history disappeared.",
        )

        assert_true(
            "Advanced RAG Platform" in history_a_after_b,
            "Conversation A original answer disappeared.",
        )

        assert_true(
            "What database am I using?" not in history_a_after_b,
            "Conversation A can see Conversation B question.",
        )

        assert_true(
            "PostgreSQL" not in history_a_after_b,
            "Conversation A can see Conversation B answer.",
        )

        print(
            "PASS: Isolation works in both directions."
        )

        # -------------------------------------------------------------------
        # TEST 6 — SUMMARY + RECENT HISTORY
        # -------------------------------------------------------------------

        print_section("TEST 6 — SUMMARY + RECENT HISTORY")

        summary_marker = (
            "User is building a production-oriented Advanced RAG system "
            "with multi-tenant security."
        )

        memory.redis.set(
            memory._summary_key(conversation_a),
            summary_marker,
        )

        combined_history = memory.build_chat_history(
            conversation_id=conversation_a,
        )

        assert_true(
            summary_marker in combined_history,
            "Conversation summary was not included.",
        )

        assert_true(
            "What is my project?" in combined_history,
            "Recent message was not included with summary.",
        )

        print(
            "PASS: Summary and recent messages are combined correctly."
        )

        # -------------------------------------------------------------------
        # TEST 7 — SUMMARY ISOLATION
        # -------------------------------------------------------------------

        print_section("TEST 7 — SUMMARY ISOLATION")

        history_b_with_a_summary = memory.build_chat_history(
            conversation_id=conversation_b,
        )

        assert_true(
            summary_marker not in history_b_with_a_summary,
            "Conversation B can see Conversation A summary.",
        )

        print(
            "PASS: Conversation A summary is isolated from Conversation B."
        )

        # -------------------------------------------------------------------
        # TEST 8 — MESSAGE LIMIT / COMPRESSION
        # -------------------------------------------------------------------

        print_section("TEST 8 — MESSAGE LIMIT / COMPRESSION")

        compression_conversation = memory.create_conversation_id()

        memory.clear(compression_conversation)

        # Avoid calling Gemini during this test.
        # Replace summary generation with a deterministic fake summary.
        generated_summary = (
            "Compressed summary of older conversation messages."
        )

        def fake_generate_summary(
            existing_summary: str,
            messages_to_summarize: list[dict],
        ) -> str:
            assert_true(
                len(messages_to_summarize) > 0,
                "Compression was triggered without old messages.",
            )

            return generated_summary

        memory._generate_summary = fake_generate_summary  # type: ignore

        try:
            # Save 6 turns = 12 Redis messages.
            # Limit is 10 messages.
            for index in range(1, 7):
                memory.save_turn(
                    conversation_id=compression_conversation,
                    question=f"Compression question {index}",
                    answer=f"Compression answer {index}",
                )

            recent_messages = memory.get_recent_messages(
                compression_conversation,
            )

            summary = memory.get_summary(
                compression_conversation,
            )

            # After compression exactly 10 recent messages should remain.
            assert_true(
                len(recent_messages) == 10,
                (
                    "Expected exactly 10 recent messages after compression, "
                    f"got {len(recent_messages)}."
                ),
            )

            assert_true(
                summary == generated_summary,
                "Compressed summary was not stored correctly.",
            )

            # First turn should have been moved into summary.
            recent_text = "\n".join(
                str(message.get("content", ""))
                for message in recent_messages
            )

            assert_true(
                "Compression question 1" not in recent_text,
                "Old question 1 was not removed from recent messages.",
            )

            assert_true(
                "Compression answer 1" not in recent_text,
                "Old answer 1 was not removed from recent messages.",
            )

            # Latest turn should still exist.
            assert_true(
                "Compression question 6" in recent_text,
                "Latest question was incorrectly removed.",
            )

            assert_true(
                "Compression answer 6" in recent_text,
                "Latest answer was incorrectly removed.",
            )

            print(
                "PASS: Message limit and compression work correctly."
            )

        finally:
            memory.clear(compression_conversation)

        # -------------------------------------------------------------------
        # TEST 9 — CLEAR CONVERSATION
        # -------------------------------------------------------------------

        print_section("TEST 9 — CLEAR CONVERSATION")

        # Add fresh data so clear() is tested independently.
        memory.save_turn(
            conversation_id=conversation_a,
            question="Temporary question before clear",
            answer="Temporary answer before clear",
        )

        memory.redis.set(
            memory._summary_key(conversation_a),
            "Temporary summary before clear",
        )

        # Verify data exists before clearing.
        assert_true(
            memory.get_recent_messages(conversation_a),
            "Conversation A should contain messages before clear.",
        )

        assert_true(
            memory.get_summary(conversation_a) != "",
            "Conversation A should contain a summary before clear.",
        )

        memory.clear(conversation_a)

        messages_after_clear = memory.get_recent_messages(
            conversation_a,
        )

        summary_after_clear = memory.get_summary(
            conversation_a,
        )

        assert_true(
            messages_after_clear == [],
            "Conversation A messages were not deleted by clear().",
        )

        assert_true(
            summary_after_clear == "",
            "Conversation A summary was not deleted by clear().",
        )

        print(
            "PASS: clear() removed messages and summary."
        )

        # -------------------------------------------------------------------
        # TEST 10 — NEW CONVERSATION STARTS EMPTY
        # -------------------------------------------------------------------

        print_section("TEST 10 — NEW CONVERSATION STARTS EMPTY")

        new_conversation = memory.create_conversation_id()

        try:
            new_history = memory.build_chat_history(
                conversation_id=new_conversation,
            )

            assert_true(
                new_history == "",
                "New conversation is not empty.",
            )

            assert_true(
                memory.get_recent_messages(new_conversation) == [],
                "New conversation unexpectedly contains messages.",
            )

            assert_true(
                memory.get_summary(new_conversation) == "",
                "New conversation unexpectedly contains summary.",
            )

            print(
                "PASS: New conversation starts with isolated empty memory."
            )

        finally:
            memory.clear(new_conversation)

        # -------------------------------------------------------------------
        # FINAL
        # -------------------------------------------------------------------

        print()
        print()
        print("=" * 75)
        print("ALL SHORT-TERM CONVERSATION MEMORY TESTS PASSED")
        print("=" * 75)

    finally:
        # -------------------------------------------------------------------
        # CLEANUP
        # -------------------------------------------------------------------

        memory.clear(conversation_a)
        memory.clear(conversation_b)


if __name__ == "__main__":
    main()