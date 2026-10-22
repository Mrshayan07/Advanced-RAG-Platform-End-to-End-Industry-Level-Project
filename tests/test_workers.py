from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.workers.tasks import ingest_document_task


def test_ingest_document_task_requires_document_id():
    with pytest.raises(
        ValueError,
        match="document_id is required",
    ):
        ingest_document_task.run(
            document_id="",
            storage_key="storage/test.txt",
            tenant_id=str(uuid4()),
        )


def test_ingest_document_task_requires_tenant_id():
    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        ingest_document_task.run(
            document_id=str(uuid4()),
            storage_key="storage/test.txt",
            tenant_id="",
        )


def test_ingest_document_task_marks_document_indexed_on_success():
    document_id = uuid4()
    tenant_id = uuid4()

    ingestion_service = MagicMock()
    ingestion_service.ingest.return_value = 7

    with (
        patch(
            "app.workers.tasks.IngestionService",
            return_value=ingestion_service,
        ),
        patch(
            "app.workers.tasks._update_document_status",
        ) as update_status,
    ):
        result = ingest_document_task.run(
            document_id=str(document_id),
            storage_key="storage/test.txt",
            tenant_id=str(tenant_id),
            original_file_name="test.txt",
            chunk_size=600,
            chunk_overlap=150,
        )

    assert result == 7

    ingestion_service.ingest.assert_called_once_with(
        storage_key="storage/test.txt",
        tenant_id=tenant_id,
        original_file_name="test.txt",
        chunk_size=600,
        chunk_overlap=150,
    )

    update_status.assert_called_once_with(
        document_id=document_id,
        status="indexed",
    )


def test_ingest_document_task_marks_document_failed_on_error():
    document_id = uuid4()
    tenant_id = uuid4()

    ingestion_service = MagicMock()
    ingestion_service.ingest.side_effect = RuntimeError(
        "ingestion failed"
    )

    with (
        patch(
            "app.workers.tasks.IngestionService",
            return_value=ingestion_service,
        ),
        patch(
            "app.workers.tasks._update_document_status",
        ) as update_status,
    ):
        with pytest.raises(
            RuntimeError,
            match="ingestion failed",
        ):
            ingest_document_task.run(
                document_id=str(document_id),
                storage_key="storage/test.txt",
                tenant_id=str(tenant_id),
                original_file_name="test.txt",
            )

    update_status.assert_called_once_with(
        document_id=document_id,
        status="failed",
    )


def test_ingest_document_task_preserves_original_error_if_status_update_fails():
    document_id = uuid4()
    tenant_id = uuid4()

    ingestion_service = MagicMock()
    ingestion_service.ingest.side_effect = RuntimeError(
        "original ingestion error"
    )

    with (
        patch(
            "app.workers.tasks.IngestionService",
            return_value=ingestion_service,
        ),
        patch(
            "app.workers.tasks._update_document_status",
            side_effect=RuntimeError(
                "database update failed"
            ),
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="original ingestion error",
        ):
            ingest_document_task.run(
                document_id=str(document_id),
                storage_key="storage/test.txt",
                tenant_id=str(tenant_id),
            )


def test_ingest_document_task_converts_ids_to_uuid():
    document_id = uuid4()
    tenant_id = uuid4()

    ingestion_service = MagicMock()
    ingestion_service.ingest.return_value = 1

    with (
        patch(
            "app.workers.tasks.IngestionService",
            return_value=ingestion_service,
        ),
        patch(
            "app.workers.tasks._update_document_status",
        ) as update_status,
    ):
        result = ingest_document_task.run(
            document_id=str(document_id),
            storage_key="storage/test.txt",
            tenant_id=str(tenant_id),
        )

    assert result == 1

    call_kwargs = ingestion_service.ingest.call_args.kwargs

    assert call_kwargs["tenant_id"] == tenant_id
    assert isinstance(call_kwargs["tenant_id"], type(tenant_id))

    status_kwargs = update_status.call_args.kwargs

    assert status_kwargs["document_id"] == document_id
    assert isinstance(
        status_kwargs["document_id"],
        type(document_id),
    )