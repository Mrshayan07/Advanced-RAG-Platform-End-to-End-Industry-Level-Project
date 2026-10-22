import asyncio
from uuid import UUID, uuid4

from app.core.security import hash_password
from app.db.models import Conversation, Message, User
from tests.conftest import TestSessionLocal


def register_and_login(
    client,
    email: str,
    password: str = "Test123!",
    tenant_name: str = "Conversation Test Tenant",
):
    register_response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "tenant_name": tenant_name,
        },
    )

    assert register_response.status_code == 201

    registration_data = register_response.json()

    login_response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}",
    }, registration_data


def create_conversation(
    user_id: str | UUID,
    tenant_id: str | UUID,
    title: str = "Test Conversation",
    messages: list[tuple[str, str]] | None = None,
):
    async def _create():
        async with TestSessionLocal() as session:
            conversation = Conversation(
                id=uuid4(),
                user_id=UUID(str(user_id)),
                tenant_id=UUID(str(tenant_id)),
                title=title,
            )

            session.add(conversation)
            await session.flush()

            if messages:
                for sequence_number, (role, content) in enumerate(
                    messages,
                    start=1,
                ):
                    session.add(
                        Message(
                            id=uuid4(),
                            conversation_id=conversation.id,
                            sequence_number=sequence_number,
                            role=role,
                            content=content,
                        )
                    )

            await session.commit()
            await session.refresh(conversation)

            return conversation.id

    return asyncio.run(_create())


def create_user_in_same_tenant(
    tenant_id: str | UUID,
    email: str,
    password: str = "Regular123!",
):
    async def _create():
        async with TestSessionLocal() as session:
            user = User(
                id=uuid4(),
                tenant_id=UUID(str(tenant_id)),
                email=email,
                password_hash=hash_password(password),
                role="user",
            )

            session.add(user)
            await session.commit()
            await session.refresh(user)

            return user.id

    return asyncio.run(_create())


def get_conversation_from_db(conversation_id: str | UUID):
    async def _get():
        async with TestSessionLocal() as session:
            conversation = await session.get(
                Conversation,
                UUID(str(conversation_id)),
            )

            return conversation

    return asyncio.run(_get())


def get_messages_from_db(conversation_id: str | UUID):
    async def _get():
        async with TestSessionLocal() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(Message)
                .where(
                    Message.conversation_id
                    == UUID(str(conversation_id))
                )
            )

            return result.scalars().all()

    return asyncio.run(_get())


def test_list_conversations_requires_authentication(client):
    response = client.get("/conversations")

    assert response.status_code == 401


def test_list_conversations_returns_empty_list_for_new_user(client):
    headers, _ = register_and_login(
        client,
        "conversation-empty@example.com",
    )

    response = client.get(
        "/conversations",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_list_conversations_returns_user_conversations(client):
    headers, user_data = register_and_login(
        client,
        "conversation-list@example.com",
    )

    conversation_id_one = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="First Conversation",
    )

    conversation_id_two = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Second Conversation",
    )

    response = client.get(
        "/conversations",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    returned_ids = {
        item["id"]
        for item in data
    }

    assert str(conversation_id_one) in returned_ids
    assert str(conversation_id_two) in returned_ids

    returned_titles = {
        item["title"]
        for item in data
    }

    assert returned_titles == {
        "First Conversation",
        "Second Conversation",
    }


def test_list_conversations_does_not_return_other_users_conversations(
    client,
):
    owner_headers, owner_data = register_and_login(
        client,
        "conversation-owner@example.com",
        tenant_name="Shared Conversation Tenant",
    )

    regular_user_email = "conversation-other-user@example.com"
    regular_user_password = "Regular123!"

    regular_user_id = create_user_in_same_tenant(
        tenant_id=owner_data["tenant_id"],
        email=regular_user_email,
        password=regular_user_password,
    )

    owner_conversation_id = create_conversation(
        user_id=owner_data["id"],
        tenant_id=owner_data["tenant_id"],
        title="Owner Conversation",
    )

    other_user_conversation_id = create_conversation(
        user_id=regular_user_id,
        tenant_id=owner_data["tenant_id"],
        title="Other User Conversation",
    )

    login_response = client.post(
        "/auth/login",
        json={
            "email": regular_user_email,
            "password": regular_user_password,
        },
    )

    assert login_response.status_code == 200

    other_user_headers = {
        "Authorization": (
            f"Bearer {login_response.json()['access_token']}"
        ),
    }

    response = client.get(
        "/conversations",
        headers=other_user_headers,
    )

    assert response.status_code == 200

    data = response.json()

    returned_ids = {
        item["id"]
        for item in data
    }

    assert str(other_user_conversation_id) in returned_ids
    assert str(owner_conversation_id) not in returned_ids

    assert len(data) == 1
    assert data[0]["title"] == "Other User Conversation"


def test_list_conversations_does_not_return_other_tenant_conversations(
    client,
):
    headers_one, user_one = register_and_login(
        client,
        "conversation-tenant-one@example.com",
        tenant_name="Conversation Tenant One",
    )

    _, user_two = register_and_login(
        client,
        "conversation-tenant-two@example.com",
        tenant_name="Conversation Tenant Two",
    )

    tenant_one_conversation = create_conversation(
        user_id=user_one["id"],
        tenant_id=user_one["tenant_id"],
        title="Tenant One Conversation",
    )

    create_conversation(
        user_id=user_two["id"],
        tenant_id=user_two["tenant_id"],
        title="Tenant Two Conversation",
    )

    response = client.get(
        "/conversations",
        headers=headers_one,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["id"] == str(tenant_one_conversation)
    assert data[0]["title"] == "Tenant One Conversation"


def test_get_conversation_requires_authentication(client):
    conversation_id = uuid4()

    response = client.get(
        f"/conversations/{conversation_id}",
    )

    assert response.status_code == 401


def test_get_conversation_returns_conversation_without_messages(
    client,
):
    headers, user_data = register_and_login(
        client,
        "conversation-get-empty@example.com",
    )

    conversation_id = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Empty Conversation",
    )

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(conversation_id)
    assert data["title"] == "Empty Conversation"
    assert "created_at" in data
    assert data["messages"] == []


def test_get_conversation_returns_messages_in_created_order(
    client,
):
    headers, user_data = register_and_login(
        client,
        "conversation-messages@example.com",
    )

    conversation_id = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Message Conversation",
        messages=[
            ("user", "Hello"),
            ("assistant", "Hello! How can I help?"),
            ("user", "Tell me about RAG."),
        ],
    )

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(conversation_id)
    assert data["title"] == "Message Conversation"

    assert len(data["messages"]) == 3

    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hello"

    assert data["messages"][1]["role"] == "assistant"
    assert (
        data["messages"][1]["content"]
        == "Hello! How can I help?"
    )

    assert data["messages"][2]["role"] == "user"
    assert (
        data["messages"][2]["content"]
        == "Tell me about RAG."
    )

    for message in data["messages"]:
        assert "id" in message
        assert "created_at" in message
        assert "citations" in message
        assert message["citations"] == []


def test_get_conversation_returns_404_for_unknown_conversation(
    client,
):
    headers, _ = register_and_login(
        client,
        "conversation-get-unknown@example.com",
    )

    conversation_id = uuid4()

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


def test_get_conversation_rejects_other_users_conversation(
    client,
):
    owner_headers, owner_data = register_and_login(
        client,
        "conversation-get-owner@example.com",
        tenant_name="Conversation Shared Tenant",
    )

    other_email = "conversation-get-other@example.com"
    other_password = "OtherUser123!"

    create_user_in_same_tenant(
        tenant_id=owner_data["tenant_id"],
        email=other_email,
        password=other_password,
    )

    conversation_id = create_conversation(
        user_id=owner_data["id"],
        tenant_id=owner_data["tenant_id"],
        title="Private Conversation",
    )

    login_response = client.post(
        "/auth/login",
        json={
            "email": other_email,
            "password": other_password,
        },
    )

    assert login_response.status_code == 200

    other_headers = {
        "Authorization": (
            f"Bearer {login_response.json()['access_token']}"
        ),
    }

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=other_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"

    owner_response = client.get(
        f"/conversations/{conversation_id}",
        headers=owner_headers,
    )

    assert owner_response.status_code == 200


def test_get_conversation_rejects_other_tenant_conversation(
    client,
):
    headers_one, user_one = register_and_login(
        client,
        "conversation-get-tenant-one@example.com",
        tenant_name="Get Tenant One",
    )

    _, user_two = register_and_login(
        client,
        "conversation-get-tenant-two@example.com",
        tenant_name="Get Tenant Two",
    )

    conversation_id = create_conversation(
        user_id=user_two["id"],
        tenant_id=user_two["tenant_id"],
        title="Other Tenant Conversation",
    )

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=headers_one,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


def test_delete_conversation_requires_authentication(client):
    conversation_id = uuid4()

    response = client.delete(
        f"/conversations/{conversation_id}",
    )

    assert response.status_code == 401


def test_delete_conversation_returns_404_for_unknown_conversation(
    client,
):
    headers, _ = register_and_login(
        client,
        "conversation-delete-unknown@example.com",
    )

    conversation_id = uuid4()

    response = client.delete(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


def test_delete_conversation_rejects_other_users_conversation(
    client,
):
    _, owner_data = register_and_login(
        client,
        "conversation-delete-owner@example.com",
        tenant_name="Delete Shared Tenant",
    )

    other_email = "conversation-delete-other@example.com"
    other_password = "OtherDelete123!"

    create_user_in_same_tenant(
        tenant_id=owner_data["tenant_id"],
        email=other_email,
        password=other_password,
    )

    conversation_id = create_conversation(
        user_id=owner_data["id"],
        tenant_id=owner_data["tenant_id"],
        title="Protected Conversation",
        messages=[
            ("user", "Private message"),
        ],
    )

    login_response = client.post(
        "/auth/login",
        json={
            "email": other_email,
            "password": other_password,
        },
    )

    assert login_response.status_code == 200

    other_headers = {
        "Authorization": (
            f"Bearer {login_response.json()['access_token']}"
        ),
    }

    response = client.delete(
        f"/conversations/{conversation_id}",
        headers=other_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"

    conversation = get_conversation_from_db(
        conversation_id
    )

    assert conversation is not None


def test_delete_conversation_rejects_other_tenant_conversation(
    client,
):
    headers_one, _ = register_and_login(
        client,
        "conversation-delete-tenant-one@example.com",
        tenant_name="Delete Tenant One",
    )

    _, user_two = register_and_login(
        client,
        "conversation-delete-tenant-two@example.com",
        tenant_name="Delete Tenant Two",
    )

    conversation_id = create_conversation(
        user_id=user_two["id"],
        tenant_id=user_two["tenant_id"],
        title="Tenant Two Protected",
    )

    response = client.delete(
        f"/conversations/{conversation_id}",
        headers=headers_one,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"

    conversation = get_conversation_from_db(
        conversation_id
    )

    assert conversation is not None


def test_delete_conversation_deletes_conversation_and_messages(
    client,
):
    headers, user_data = register_and_login(
        client,
        "conversation-delete-success@example.com",
    )

    conversation_id = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Conversation To Delete",
        messages=[
            ("user", "First message"),
            ("assistant", "Second message"),
            ("user", "Third message"),
        ],
    )

    messages_before = get_messages_from_db(
        conversation_id
    )

    assert len(messages_before) == 3

    response = client.delete(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 204
    assert response.content == b""

    conversation_after = get_conversation_from_db(
        conversation_id
    )

    assert conversation_after is None

    messages_after = get_messages_from_db(
        conversation_id
    )

    assert messages_after == []


def test_delete_conversation_removes_only_target_conversation(
    client,
):
    headers, user_data = register_and_login(
        client,
        "conversation-delete-isolation@example.com",
    )

    conversation_one = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Keep This Conversation",
        messages=[
            ("user", "Keep this message"),
        ],
    )

    conversation_two = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Delete This Conversation",
        messages=[
            ("user", "Delete this message"),
        ],
    )

    response = client.delete(
        f"/conversations/{conversation_two}",
        headers=headers,
    )

    assert response.status_code == 204

    deleted_conversation = get_conversation_from_db(
        conversation_two
    )

    remaining_conversation = get_conversation_from_db(
        conversation_one
    )

    deleted_messages = get_messages_from_db(
        conversation_two
    )

    remaining_messages = get_messages_from_db(
        conversation_one
    )

    assert deleted_conversation is None
    assert deleted_messages == []

    assert remaining_conversation is not None
    assert (
        remaining_conversation.title
        == "Keep This Conversation"
    )

    assert len(remaining_messages) == 1
    assert remaining_messages[0].content == "Keep this message"


def test_get_conversation_response_contains_expected_schema(
    client,
):
    headers, user_data = register_and_login(
        client,
        "conversation-schema@example.com",
    )

    conversation_id = create_conversation(
        user_id=user_data["id"],
        tenant_id=user_data["tenant_id"],
        title="Schema Test",
        messages=[
            ("user", "Schema message"),
        ],
    )

    response = client.get(
        f"/conversations/{conversation_id}",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "id",
        "title",
        "created_at",
        "messages",
    }

    assert isinstance(data["id"], str)
    assert isinstance(data["title"], str)
    assert isinstance(data["created_at"], str)
    assert isinstance(data["messages"], list)

    assert len(data["messages"]) == 1

    message = data["messages"][0]

    assert set(message.keys()) == {
        "id",
        "role",
        "content",
        "citations",
        "created_at",
    }

    assert isinstance(message["id"], str)
    assert isinstance(message["role"], str)
    assert isinstance(message["content"], str)
    assert isinstance(message["citations"], list)
    assert message["citations"] == []
    assert isinstance(message["created_at"], str)