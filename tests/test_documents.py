from io import BytesIO
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4


def register_and_login(
    client,
    email: str,
    password: str = "Test123!",
    tenant_name: str = "Document Test Tenant",
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
    }


def create_text_upload(
    filename: str = "test.txt",
    content: bytes = b"Hello from document testing.",
    content_type: str = "text/plain",
):
    return {
        "file": (
            filename,
            BytesIO(content),
            content_type,
        )
    }


def test_list_documents_requires_authentication(client):
    response = client.get("/documents")

    assert response.status_code == 401


def test_list_documents_returns_empty_list_for_new_tenant(client):
    headers = register_and_login(
        client,
        "documents-empty@example.com",
    )

    response = client.get(
        "/documents",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_tenant_documents_only(client):
    headers_one = register_and_login(
        client,
        "documents-tenant-one@example.com",
        tenant_name="Tenant One",
    )

    headers_two = register_and_login(
        client,
        "documents-tenant-two@example.com",
        tenant_name="Tenant Two",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/tenant-one.txt", "checksum-one"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers_one,
            files=create_text_upload(
                filename="tenant-one.txt",
            ),
        )

    assert upload_response.status_code == 201

    response_one = client.get(
        "/documents",
        headers=headers_one,
    )

    assert response_one.status_code == 200

    documents_one = response_one.json()

    assert len(documents_one) == 1
    assert documents_one[0]["name"] == "tenant-one.txt"
    assert documents_one[0]["status"] == "processing"

    response_two = client.get(
        "/documents",
        headers=headers_two,
    )

    assert response_two.status_code == 200
    assert response_two.json() == []


def test_upload_requires_authentication(client):
    response = client.post(
        "/documents/upload",
        files=create_text_upload(),
    )

    assert response.status_code == 401


def test_upload_rejects_unsupported_file_type(client):
    headers = register_and_login(
        client,
        "documents-unsupported@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        files=create_text_upload(
            filename="test.jpg",
            content_type="image/jpeg",
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"


def test_upload_rejects_chunk_size_below_minimum(client):
    headers = register_and_login(
        client,
        "documents-small-chunk@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "299",
            "chunk_overlap": "100",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_size must be between 300 and 2000"
    )


def test_upload_rejects_chunk_size_above_maximum(client):
    headers = register_and_login(
        client,
        "documents-large-chunk@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "2001",
            "chunk_overlap": "100",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_size must be between 300 and 2000"
    )


def test_upload_rejects_negative_chunk_overlap(client):
    headers = register_and_login(
        client,
        "documents-negative-overlap@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "600",
            "chunk_overlap": "-1",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_overlap must be between 0 and 500"
    )


def test_upload_rejects_chunk_overlap_above_maximum(client):
    headers = register_and_login(
        client,
        "documents-large-overlap@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "2000",
            "chunk_overlap": "501",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_overlap must be between 0 and 500"
    )


def test_upload_rejects_overlap_equal_to_chunk_size(client):
    headers = register_and_login(
        client,
        "documents-equal-overlap@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "600",
            "chunk_overlap": "600",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_overlap must be smaller than chunk_size"
    )


def test_upload_rejects_overlap_greater_than_chunk_size(client):
    headers = register_and_login(
        client,
        "documents-greater-overlap@example.com",
    )

    response = client.post(
        "/documents/upload",
        headers=headers,
        data={
            "chunk_size": "600",
            "chunk_overlap": "700",
        },
        files=create_text_upload(),
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "chunk_overlap must be smaller than chunk_size"
    )


def test_upload_queues_background_ingestion(client):
    headers = register_and_login(
        client,
        "documents-queue@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/test.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(),
        )

    assert response.status_code == 201

    data = response.json()

    assert UUID(data["id"])
    assert data["name"] == "test.txt"
    assert data["mime_type"] == "text/plain"
    assert data["status"] == "processing"
    assert UUID(data["tenant_id"])

    fake_task.delay.assert_called_once()

    call_kwargs = fake_task.delay.call_args.kwargs

    assert call_kwargs["document_id"] == data["id"]
    assert call_kwargs["storage_key"] == "uploads/test.txt"
    assert call_kwargs["tenant_id"] == data["tenant_id"]
    assert call_kwargs["original_file_name"] == "test.txt"
    assert call_kwargs["chunk_size"] == 600
    assert call_kwargs["chunk_overlap"] == 150


def test_upload_uses_default_chunk_parameters(client):
    headers = register_and_login(
        client,
        "documents-defaults@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/defaults.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="defaults.txt",
            ),
        )

    assert response.status_code == 201
    assert response.json()["status"] == "processing"

    fake_task.delay.assert_called_once()

    call_kwargs = fake_task.delay.call_args.kwargs

    assert call_kwargs["chunk_size"] == 600
    assert call_kwargs["chunk_overlap"] == 150


def test_upload_passes_custom_chunk_parameters_to_task(client):
    headers = register_and_login(
        client,
        "documents-custom-chunks@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/custom.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        response = client.post(
            "/documents/upload",
            headers=headers,
            data={
                "chunk_size": "800",
                "chunk_overlap": "200",
            },
            files=create_text_upload(
                filename="custom.txt",
            ),
        )

    assert response.status_code == 201
    assert response.json()["status"] == "processing"

    fake_task.delay.assert_called_once()

    call_kwargs = fake_task.delay.call_args.kwargs

    assert call_kwargs["chunk_size"] == 800
    assert call_kwargs["chunk_overlap"] == 200


def test_upload_marks_document_failed_when_queueing_fails(client):
    headers = register_and_login(
        client,
        "documents-queue-failure@example.com",
    )

    fake_task = MagicMock()
    fake_task.delay.side_effect = RuntimeError(
        "Redis unavailable"
    )

    async def fake_save_upload(file):
        return "uploads/failure.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="failure.txt",
            ),
        )

    assert response.status_code == 500
    assert (
        response.json()["detail"]
        == "Failed to queue document ingestion"
    )

    list_response = client.get(
        "/documents",
        headers=headers,
    )

    assert list_response.status_code == 200

    data = list_response.json()

    assert len(data) == 1
    assert data[0]["name"] == "failure.txt"
    assert data[0]["status"] == "failed"


def test_upload_passes_current_tenant_to_create_document(client):
    headers = register_and_login(
        client,
        "documents-tenant-check@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/tenant-check.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="tenant-check.txt",
            ),
        )

    assert response.status_code == 201

    document = response.json()

    assert UUID(document["tenant_id"])

    list_response = client.get(
        "/documents",
        headers=headers,
    )

    assert list_response.status_code == 200

    listed_documents = list_response.json()

    assert len(listed_documents) == 1

    assert (
        listed_documents[0]["tenant_id"]
        == document["tenant_id"]
    )


def test_delete_requires_authentication(client):
    document_id = uuid4()

    response = client.delete(
        f"/documents/{document_id}",
    )

    assert response.status_code == 401


def test_delete_returns_404_for_unknown_document(client):
    headers = register_and_login(
        client,
        "documents-delete-unknown@example.com",
    )

    document_id = uuid4()

    response = client.delete(
        f"/documents/{document_id}",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_rejects_document_from_other_tenant(client):
    headers_one = register_and_login(
        client,
        "documents-delete-owner@example.com",
        tenant_name="Owner Tenant",
    )

    headers_two = register_and_login(
        client,
        "documents-delete-other@example.com",
        tenant_name="Other Tenant",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/delete-owner.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers_one,
            files=create_text_upload(
                filename="delete-owner.txt",
            ),
        )

    assert upload_response.status_code == 201

    document_id = upload_response.json()["id"]

    response = client.delete(
        f"/documents/{document_id}",
        headers=headers_two,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_removes_qdrant_vectors_and_database_document(client):
    headers = register_and_login(
        client,
        "documents-delete-success@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/delete-success.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="delete-success.txt",
            ),
        )

    assert upload_response.status_code == 201

    document_id = upload_response.json()["id"]
    tenant_id = UUID(upload_response.json()["tenant_id"])

    qdrant_service = MagicMock()

    with patch(
        "app.api.documents.QdrantService",
        return_value=qdrant_service,
    ):
        response = client.delete(
            f"/documents/{document_id}",
            headers=headers,
        )

    assert response.status_code == 204
    assert response.content == b""

    qdrant_service.delete_documents_by_source.assert_called_once_with(
        source="uploads/delete-success.txt",
        tenant_id=tenant_id,
    )

    list_response = client.get(
        "/documents",
        headers=headers,
    )

    assert list_response.status_code == 200
    assert list_response.json() == []


def test_delete_removes_stored_file_when_it_exists(client):
    headers = register_and_login(
        client,
        "documents-delete-file@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/delete-file.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="delete-file.txt",
            ),
        )

    assert upload_response.status_code == 201

    document_id = upload_response.json()["id"]

    qdrant_service = MagicMock()
    fake_path = MagicMock()
    fake_path.exists.return_value = True

    with (
        patch(
            "app.api.documents.QdrantService",
            return_value=qdrant_service,
        ),
        patch(
            "app.api.documents.Path",
            return_value=fake_path,
        ),
    ):
        response = client.delete(
            f"/documents/{document_id}",
            headers=headers,
        )

    assert response.status_code == 204

    fake_path.exists.assert_called_once()
    fake_path.unlink.assert_called_once()


def test_delete_handles_qdrant_failure(client):
    headers = register_and_login(
        client,
        "documents-delete-qdrant-failure@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/qdrant-failure.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="qdrant-failure.txt",
            ),
        )

    assert upload_response.status_code == 201

    document_id = upload_response.json()["id"]

    qdrant_service = MagicMock()
    qdrant_service.delete_documents_by_source.side_effect = RuntimeError(
        "Qdrant unavailable"
    )

    with patch(
        "app.api.documents.QdrantService",
        return_value=qdrant_service,
    ):
        response = client.delete(
            f"/documents/{document_id}",
            headers=headers,
        )

    assert response.status_code == 500
    assert (
        "Failed to delete document vectors"
        in response.json()["detail"]
    )


def test_delete_handles_file_deletion_failure(client):
    headers = register_and_login(
        client,
        "documents-delete-file-failure@example.com",
    )

    fake_task = MagicMock()

    async def fake_save_upload(file):
        return "uploads/file-failure.txt", "checksum"

    with (
        patch(
            "app.api.documents.save_upload",
            side_effect=fake_save_upload,
        ),
        patch(
            "app.api.documents.ingest_document_task",
            fake_task,
        ),
    ):
        upload_response = client.post(
            "/documents/upload",
            headers=headers,
            files=create_text_upload(
                filename="file-failure.txt",
            ),
        )

    assert upload_response.status_code == 201

    document_id = upload_response.json()["id"]

    qdrant_service = MagicMock()

    fake_path = MagicMock()
    fake_path.exists.return_value = True
    fake_path.unlink.side_effect = OSError(
        "Permission denied"
    )

    with (
        patch(
            "app.api.documents.QdrantService",
            return_value=qdrant_service,
        ),
        patch(
            "app.api.documents.Path",
            return_value=fake_path,
        ),
    ):
        response = client.delete(
            f"/documents/{document_id}",
            headers=headers,
        )

    assert response.status_code == 500
    assert (
        "Failed to delete stored document"
        in response.json()["detail"]
    )

    qdrant_service.delete_documents_by_source.assert_called_once()