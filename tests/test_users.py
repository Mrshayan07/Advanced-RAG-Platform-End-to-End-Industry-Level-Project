from fastapi.testclient import TestClient


def register_admin(
    client: TestClient,
    email: str,
    password: str,
    tenant_name: str,
):
    response = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "tenant_name": tenant_name,
        },
    )

    assert response.status_code == 201

    return response.json()


def login(
    client: TestClient,
    email: str,
    password: str,
):
    response = client.post(
        "/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert response.status_code == 200

    return response.json()["access_token"]


def auth_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
    }


def test_admin_can_create_user(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_1@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 1",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_1@example.com",
            "password": "UserPassword123!",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "normal_user_1@example.com"
    assert data["role"] == "user"
    assert data["tenant_id"] == admin["tenant_id"]
    assert "id" in data


def test_created_user_can_login_as_user(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_2@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 2",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    create_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_2@example.com",
            "password": "UserPassword123!",
        },
    )

    assert create_response.status_code == 201

    user_token = login(
        client=client,
        email="normal_user_2@example.com",
        password="UserPassword123!",
    )

    me_response = client.get(
        "/auth/me",
        headers=auth_headers(user_token),
    )

    assert me_response.status_code == 200

    data = me_response.json()

    assert data["role"] == "user"
    assert data["tenant_id"] == admin["tenant_id"]


def test_normal_user_cannot_create_user(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_3@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 3",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    create_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_3@example.com",
            "password": "UserPassword123!",
        },
    )

    assert create_response.status_code == 201

    user_token = login(
        client=client,
        email="normal_user_3@example.com",
        password="UserPassword123!",
    )

    response = client.post(
        "/users",
        headers=auth_headers(user_token),
        json={
            "email": "another_user_3@example.com",
            "password": "AnotherPassword123!",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Insufficient permissions"


def test_normal_user_cannot_list_users(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_4@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 4",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    create_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_4@example.com",
            "password": "UserPassword123!",
        },
    )

    assert create_response.status_code == 201

    user_token = login(
        client=client,
        email="normal_user_4@example.com",
        password="UserPassword123!",
    )

    response = client.get(
        "/users",
        headers=auth_headers(user_token),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Insufficient permissions"


def test_admin_can_list_users_in_own_tenant(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_5@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 5",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    first_user_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_5a@example.com",
            "password": "UserPassword123!",
        },
    )

    second_user_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "normal_user_5b@example.com",
            "password": "UserPassword123!",
        },
    )

    assert first_user_response.status_code == 201
    assert second_user_response.status_code == 201

    response = client.get(
        "/users",
        headers=auth_headers(admin_token),
    )

    assert response.status_code == 200

    users = response.json()

    assert len(users) == 3

    emails = {user["email"] for user in users}

    assert emails == {
        "admin_users_5@example.com",
        "normal_user_5a@example.com",
        "normal_user_5b@example.com",
    }

    assert all(
        user["tenant_id"] == admin["tenant_id"]
        for user in users
    )


def test_duplicate_email_is_rejected(client: TestClient):
    admin = register_admin(
        client=client,
        email="admin_users_6@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant Users 6",
    )

    admin_token = login(
        client=client,
        email=admin["email"],
        password="AdminPassword123!",
    )

    first_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "duplicate_user@example.com",
            "password": "UserPassword123!",
        },
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/users",
        headers=auth_headers(admin_token),
        json={
            "email": "duplicate_user@example.com",
            "password": "DifferentPassword123!",
        },
    )

    assert second_response.status_code == 409
    assert (
        second_response.json()["detail"]
        == "User with this email already exists"
    )


def test_users_are_isolated_between_tenants(client: TestClient):
    tenant_a_admin = register_admin(
        client=client,
        email="admin_tenant_a@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant A",
    )

    tenant_b_admin = register_admin(
        client=client,
        email="admin_tenant_b@example.com",
        password="AdminPassword123!",
        tenant_name="Tenant B",
    )

    tenant_a_token = login(
        client=client,
        email="admin_tenant_a@example.com",
        password="AdminPassword123!",
    )

    tenant_b_token = login(
        client=client,
        email="admin_tenant_b@example.com",
        password="AdminPassword123!",
    )

    response_a = client.post(
        "/users",
        headers=auth_headers(tenant_a_token),
        json={
            "email": "tenant_a_user@example.com",
            "password": "UserPassword123!",
        },
    )

    response_b = client.post(
        "/users",
        headers=auth_headers(tenant_b_token),
        json={
            "email": "tenant_b_user@example.com",
            "password": "UserPassword123!",
        },
    )

    assert response_a.status_code == 201
    assert response_b.status_code == 201

    users_a_response = client.get(
        "/users",
        headers=auth_headers(tenant_a_token),
    )

    users_b_response = client.get(
        "/users",
        headers=auth_headers(tenant_b_token),
    )

    assert users_a_response.status_code == 200
    assert users_b_response.status_code == 200

    users_a = users_a_response.json()
    users_b = users_b_response.json()

    emails_a = {user["email"] for user in users_a}
    emails_b = {user["email"] for user in users_b}

    assert emails_a == {
        "admin_tenant_a@example.com",
        "tenant_a_user@example.com",
    }

    assert emails_b == {
        "admin_tenant_b@example.com",
        "tenant_b_user@example.com",
    }

    assert emails_a.isdisjoint(emails_b)

    assert all(
        user["tenant_id"] == tenant_a_admin["tenant_id"]
        for user in users_a
    )

    assert all(
        user["tenant_id"] == tenant_b_admin["tenant_id"]
        for user in users_b
    )