
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api import rag


def register_and_login(
    client,
    email: str,
    password: str = "Test123!",
    tenant_name: str = "RAG Test Tenant",
):
    register_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "tenant_name": tenant_name,
        },
    )

    assert register_response.status_code in (200, 201), (
        f"Registration failed: "
        f"{register_response.status_code} "
        f"{register_response.text}"
    )

    login_response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert login_response.status_code == 200, (
        f"Login failed: "
        f"{login_response.status_code} "
        f"{login_response.text}"
    )

    token = login_response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}"
    }


def test_rag_query_requires_authentication(client):
    response = client.post(
        "/rag/query",
        json={
            "question": "What technologies does Shayan use?"
        },
    )

    assert response.status_code == 401


def test_rag_query_rejects_empty_question(client):
    headers = register_and_login(
        client,
        "rag-empty@example.com",
    )

    response = client.post(
        "/rag/query",
        headers=headers,
        json={"question": ""},
    )

    assert response.status_code == 422


def test_rag_query_rejects_missing_question(client):
    headers = register_and_login(
        client,
        "rag-missing@example.com",
    )

    response = client.post(
        "/rag/query",
        headers=headers,
        json={},
    )

    assert response.status_code == 422


def test_rag_query_success(client):
    headers = register_and_login(
        client,
        "rag-success@example.com",
    )

    mock_graph_result = {
        "answer": "Shayan works with React and FastAPI.",
        "citations": [],
    }

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value=mock_graph_result,
        ) as run_graph_mock,
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ) as build_history_mock,
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ) as save_turn_mock,
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ) as load_memory_mock,
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ) as update_title_mock,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={
                "question": "What technologies does Shayan use?"
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Shayan works with React and FastAPI."
    assert "conversation_id" in body
    assert body["conversation_id"]
    assert body["citations"] == []

    build_history_mock.assert_called_once()
    load_memory_mock.assert_called_once()
    run_graph_mock.assert_called_once()
    save_turn_mock.assert_called_once()
    update_title_mock.assert_called_once()
    memory_manager.process_user_message.assert_called_once()


def test_rag_query_passes_question_to_graph(client):
    headers = register_and_login(
        client,
        "rag-question@example.com",
    )

    question = "What is Shayan's primary professional role?"

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Full Stack Developer.",
                "citations": [],
            },
        ) as run_graph_mock,
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": question},
        )

    assert response.status_code == 200

    call_kwargs = run_graph_mock.call_args.kwargs

    assert call_kwargs["question"] == question


def test_rag_query_passes_chat_history_to_graph(client):
    headers = register_and_login(
        client,
        "rag-history@example.com",
    )

    chat_history = (
        "User: What frontend technologies does Shayan use?\n"
        "Assistant: Shayan works with React."
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "He also works with FastAPI.",
                "citations": [],
            },
        ) as run_graph_mock,
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value=chat_history,
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={
                "question": "What backend technologies?"
            },
        )

    assert response.status_code == 200

    call_kwargs = run_graph_mock.call_args.kwargs

    assert call_kwargs["chat_history"] == chat_history


def test_rag_query_loads_user_memory(client):
    headers = register_and_login(
        client,
        "rag-memory@example.com",
    )

    user_memory = [
        {
            "key": "name",
            "value": "Shayan Javed",
            "memory_type": "profile",
        },
        {
            "key": "university",
            "value": "SMIU",
            "memory_type": "education",
        },
    ]

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Answer.",
                "citations": [],
            },
        ) as run_graph_mock,
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=user_memory,
        ) as load_memory_mock,
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "Tell me about Shayan."},
        )

    assert response.status_code == 200

    load_memory_mock.assert_called_once()

    call_kwargs = run_graph_mock.call_args.kwargs

    assert call_kwargs["user_memory"] == user_memory


def test_rag_query_builds_chat_history_from_conversation(client):
    headers = register_and_login(
        client,
        "rag-redis-history@example.com",
    )

    history = (
        "User: Previous question\n"
        "Assistant: Previous answer"
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "New answer.",
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value=history,
        ) as build_history_mock,
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "Follow-up question"},
        )

    assert response.status_code == 200

    build_history_mock.assert_called_once()

    conversation_id = (
        build_history_mock.call_args.kwargs["conversation_id"]
    )

    assert conversation_id is not None
    assert isinstance(conversation_id, str)


def test_rag_query_saves_turn_to_redis(client):
    headers = register_and_login(
        client,
        "rag-redis-save@example.com",
    )

    question = "What technologies does Shayan use?"
    answer = "Shayan uses React and FastAPI."

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": answer,
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ) as save_turn_mock,
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": question},
        )

    assert response.status_code == 200

    save_turn_mock.assert_called_once()

    call_kwargs = save_turn_mock.call_args.kwargs

    assert call_kwargs["question"] == question
    assert call_kwargs["answer"] == answer
    assert call_kwargs["conversation_id"]


def test_rag_query_creates_conversation(client):
    headers = register_and_login(
        client,
        "rag-new-conversation@example.com",
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Answer.",
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ) as update_title_mock,
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "New conversation question"},
        )

    assert response.status_code == 200

    conversation_id = response.json()["conversation_id"]

    assert conversation_id

    update_title_mock.assert_called_once()


def test_rag_query_reuses_existing_conversation(client):
    headers = register_and_login(
        client,
        "rag-existing-conversation@example.com",
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Answer.",
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        first_response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "First question"},
        )

        assert first_response.status_code == 200

        conversation_id = first_response.json()["conversation_id"]

        second_response = client.post(
            "/rag/query",
            headers=headers,
            json={
                "conversation_id": conversation_id,
                "question": "Second question",
            },
        )

    assert second_response.status_code == 200

    assert (
        second_response.json()["conversation_id"]
        == conversation_id
    )


def test_rag_query_rejects_nonexistent_conversation(client):
    headers = register_and_login(
        client,
        "rag-nonexistent-conversation@example.com",
    )

    nonexistent_id = str(uuid4())

    response = client.post(
        "/rag/query",
        headers=headers,
        json={
            "conversation_id": nonexistent_id,
            "question": "Question",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_rag_query_rejects_other_users_conversation(client):
    headers_a = register_and_login(
        client,
        "rag-owner-a@example.com",
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Answer.",
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        first_response = client.post(
            "/rag/query",
            headers=headers_a,
            json={"question": "Owner A question"},
        )

    assert first_response.status_code == 200

    conversation_id = first_response.json()["conversation_id"]

    headers_b = register_and_login(
        client,
        "rag-owner-b@example.com",
    )

    response = client.post(
        "/rag/query",
        headers=headers_b,
        json={
            "conversation_id": conversation_id,
            "question": "Owner B question",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_rag_query_returns_citations(client):
    headers = register_and_login(
        client,
        "rag-citations@example.com",
    )

    citations = [
        {
            "citation_id": 1,
            "file_name": "resume.pdf",
            "page": "1",
            "chunk_index": 0,
            "source": "resume.pdf",
        }
    ]

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Shayan works with React.",
                "citations": citations,
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "What frontend technology?"},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Shayan works with React."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["citation_id"] == 1


def test_rag_query_handles_missing_answer(client):
    headers = register_and_login(
        client,
        "rag-missing-answer@example.com",
    )

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={"citations": []},
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": "Question"},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == ""
    assert body["citations"] == []


def test_rag_query_processes_user_message_for_memory(client):
    headers = register_and_login(
        client,
        "rag-memory-manager@example.com",
    )

    question = "My name is Shayan and I study at SMIU."

    with (
        patch(
            "app.api.rag.run_rag_graph",
            return_value={
                "answer": "Thanks.",
                "citations": [],
            },
        ),
        patch.object(
            rag.conversation_memory,
            "build_chat_history",
            return_value="",
        ),
        patch.object(
            rag.conversation_memory,
            "save_turn",
        ),
        patch(
            "app.api.rag.load_user_memory",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.api.rag.update_conversation_title",
            new_callable=AsyncMock,
        ),
        patch(
            "app.api.rag.MemoryManager"
        ) as memory_manager_class,
    ):
        memory_manager = memory_manager_class.return_value
        memory_manager.process_user_message = AsyncMock()

        response = client.post(
            "/rag/query",
            headers=headers,
            json={"question": question},
        )

    assert response.status_code == 200

    memory_manager.process_user_message.assert_called_once()

    call_kwargs = (
        memory_manager.process_user_message.call_args.kwargs
    )

    assert call_kwargs["tenant_id"]
    assert call_kwargs["user_id"]
    assert call_kwargs["user_message"] == question


def test_run_rag_graph_passes_tenant_user_and_role_to_graph():
    fake_graph = MagicMock()

    fake_graph.invoke.return_value = {
        "answer": "Answer.",
        "citations": [],
    }

    with patch(
        "app.api.rag.get_rag_graph",
        return_value=fake_graph,
    ):
        result = rag.run_rag_graph(
            question="Question",
            chat_history="History",
            user_memory=[],
            tenant_id="tenant-123",
            user_id="user-123",
            role="user",
        )

    assert result["answer"] == "Answer."

    fake_graph.invoke.assert_called_once()

    args, kwargs = fake_graph.invoke.call_args

    state = args[0]

    assert state["question"] == "Question"
    assert state["chat_history"] == "History"
    assert state["user_memory"] == []
    assert state["tenant_id"] == "tenant-123"

    assert kwargs["config"]["tags"] == [
        "rag-api",
        "production-rag",
    ]

    assert kwargs["config"]["metadata"] == {
        "user_id": "user-123",
        "tenant_id": "tenant-123",
        "role": "user",
    }


@pytest.mark.asyncio
async def test_update_conversation_title_creates_title():
    db = AsyncMock()

    conversation = MagicMock()
    conversation.title = "New Conversation"

    await rag.update_conversation_title(
        db=db,
        conversation=conversation,
        question="  What   technologies does Shayan use?  ",
    )

    assert conversation.title == (
        "What technologies does Shayan use?"
    )

    assert db.commit.await_count == 1
    assert db.refresh.await_count == 1


@pytest.mark.asyncio
async def test_update_conversation_title_preserves_existing_title():
    db = AsyncMock()

    conversation = MagicMock()
    conversation.title = "My Existing Conversation"

    await rag.update_conversation_title(
        db=db,
        conversation=conversation,
        question="New question",
    )

    assert conversation.title == "My Existing Conversation"

    db.commit.assert_not_awaited()
    db.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_conversation_title_ignores_empty_question():
    db = AsyncMock()

    conversation = MagicMock()
    conversation.title = "New Conversation"

    await rag.update_conversation_title(
        db=db,
        conversation=conversation,
        question="   ",
    )

    assert conversation.title == "New Conversation"

    db.commit.assert_not_awaited()
    db.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_postgres_turn_creates_user_and_assistant_messages():
    db = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()

    conversation_id = uuid4()

    conversation = MagicMock()
    conversation.id = conversation_id

    conversation_result = MagicMock()
    conversation_result.scalar_one_or_none.return_value = conversation

    max_sequence_result = MagicMock()
    max_sequence_result.scalar_one.return_value = 0

    db.execute.side_effect = [
        conversation_result,
        max_sequence_result,
    ]

    await rag.save_postgres_turn(
        db=db,
        conversation_id=conversation_id,
        question="User question",
        answer="Assistant answer",
    )

    assert db.execute.await_count == 2
    assert db.add.call_count == 2

    first_message = db.add.call_args_list[0].args[0]
    second_message = db.add.call_args_list[1].args[0]

    assert first_message.conversation_id == conversation_id
    assert first_message.sequence_number == 1
    assert first_message.role == "user"
    assert first_message.content == "User question"
    assert first_message.citations == []

    assert second_message.conversation_id == conversation_id
    assert second_message.sequence_number == 2
    assert second_message.role == "assistant"
    assert second_message.content == "Assistant answer"
    assert second_message.citations == []

    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_postgres_turn_persists_assistant_citations():
    db = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()

    conversation_id = uuid4()

    conversation = MagicMock()
    conversation.id = conversation_id

    conversation_result = MagicMock()
    conversation_result.scalar_one_or_none.return_value = conversation

    max_sequence_result = MagicMock()
    max_sequence_result.scalar_one.return_value = 4

    db.execute.side_effect = [
        conversation_result,
        max_sequence_result,
    ]

    citations = [
        {
            "citation_id": 1,
            "file_name": "resume.pdf",
            "page": "1",
            "chunk_index": 2,
            "source": "resume.pdf",
        },
        {
            "citation_id": 2,
            "file_name": "projects.pdf",
            "page": "3",
            "chunk_index": 5,
            "source": "projects.pdf",
        },
    ]

    await rag.save_postgres_turn(
        db=db,
        conversation_id=conversation_id,
        question="What projects did Shayan build?",
        answer="Shayan built several AI projects.",
        citations=citations,
    )

    assert db.execute.await_count == 2
    assert db.add.call_count == 2

    user_message = db.add.call_args_list[0].args[0]
    assistant_message = db.add.call_args_list[1].args[0]

    assert user_message.conversation_id == conversation_id
    assert user_message.sequence_number == 5
    assert user_message.role == "user"
    assert user_message.citations == []

    assert assistant_message.conversation_id == conversation_id
    assert assistant_message.sequence_number == 6
    assert assistant_message.role == "assistant"
    assert assistant_message.citations == citations

    db.commit.assert_awaited_once()
