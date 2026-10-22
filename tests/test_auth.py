
from uuid import uuid4

from app.core.security import hash_password
from app.db.models import User


def test_register_user_success(client):
    payload = {
        "email": "register_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Register Test Corp",
    }

    response = client.post("/auth/register", json=payload)

    assert response.status_code == 201

    data = response.json()

    assert "id" in data
    assert data["email"] == payload["email"]
    assert "tenant_id" in data
    assert data["role"] == "admin"


def test_register_duplicate_email(client):
    payload = {
        "email": "duplicate_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Duplicate Test Corp",
    }

    first_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert second_response.status_code == 409

    data = second_response.json()

    assert data["detail"] == "User with this email already exists"


def test_login_success(client):
    payload = {
        "email": "login_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Login Test Corp",
    }

    register_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": payload["email"],
            "password": payload["password"],
        },
    )

    assert login_response.status_code == 200

    data = login_response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 0


def test_login_invalid_password(client):
    payload = {
        "email": "invalid_password_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Invalid Password Corp",
    }

    register_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        json={
            "email": payload["email"],
            "password": "WrongPassword123!",
        },
    )

    assert login_response.status_code == 401

    data = login_response.json()

    assert data["detail"] == "Invalid email or password"


def test_login_nonexistent_user(client):
    response = client.post(
        "/auth/login",
        json={
            "email": "does_not_exist@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 401

    data = response.json()

    assert data["detail"] == "Invalid email or password"


def test_get_me_authenticated(client):
    payload = {
        "email": "me_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Me Test Corp",
    }

    register_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert register_response.status_code == 201

    registered_user = register_response.json()

    login_response = client.post(
        "/auth/login",
        json={
            "email": payload["email"],
            "password": payload["password"],
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    me_response = client.get(
        "/auth/me",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert me_response.status_code == 200

    data = me_response.json()

    assert data["user_id"] == registered_user["id"]
    assert data["tenant_id"] == registered_user["tenant_id"]
    assert data["role"] == registered_user["role"]


def test_get_me_without_token(client):
    response = client.get("/auth/me")

    assert response.status_code in [401, 403]


def test_tenant_access_authenticated(client):
    payload = {
        "email": "tenant_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Tenant Test Corp",
    }

    register_response = client.post(
        "/auth/register",
        json=payload,
    )

    assert register_response.status_code == 201

    registered_user = register_response.json()

    login_response = client.post(
        "/auth/login",
        json={
            "email": payload["email"],
            "password": payload["password"],
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    response = client.get(
        "/auth/tenant",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["tenant_id"] == registered_user["tenant_id"]
    assert data["message"] == "Tenant access granted"


def test_admin_only_rejects_non_admin(client):
    """
    Create a tenant with the normal registration endpoint.
    The first user is intentionally an admin.

    Then create a second user in the same tenant with role='user'.
    Authenticate as that regular user and verify admin-only access
    is rejected.
    """

    admin_payload = {
        "email": "admin_test@example.com",
        "password": "SecurePassword123!",
        "tenant_name": "Admin Authorization Test Corp",
    }

    register_response = client.post(
        "/auth/register",
        json=admin_payload,
    )

    assert register_response.status_code == 201

    admin_data = register_response.json()

    assert admin_data["role"] == "admin"

    regular_user_email = "regular_user_test@example.com"
    regular_user_password = "RegularPassword123!"

    # Import the same test session used by the application's dependency
    # override.
    from tests.conftest import TestSessionLocal

    async def create_regular_user():
        async with TestSessionLocal() as session:
            regular_user = User(
                id=uuid4(),
                tenant_id=admin_data["tenant_id"],
                email=regular_user_email,
                password_hash=hash_password(regular_user_password),
                role="user",
            )

            session.add(regular_user)
            await session.commit()

    import asyncio

    asyncio.run(create_regular_user())

    login_response = client.post(
        "/auth/login",
        json={
            "email": regular_user_email,
            "password": regular_user_password,
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    response = client.get(
        "/auth/admin-only",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 403

    data = response.json()

    assert data["detail"] == "Insufficient permissions"
