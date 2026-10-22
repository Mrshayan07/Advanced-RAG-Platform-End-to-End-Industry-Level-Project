from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.models import Document
from app.services.ingestion import IngestionService
from app.workers.celery_app import celery_app


async def _update_document_status(
    document_id: UUID,
    status: str,
) -> None:
    """Update document status using a fresh async DB engine."""

    settings = get_settings()

    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )

    AsyncSessionFactory = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    try:
        async with AsyncSessionFactory() as db:
            document = await db.get(Document, document_id)

            if document is None:
                raise ValueError(
                    f"Document not found: {document_id}"
                )

            document.status = status

            await db.commit()

    finally:
        await engine.dispose()


@celery_app.task(
    bind=True,
    name="advanced_rag.ingest_document",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
)
def ingest_document_task(
    self,
    document_id: str,
    storage_key: str,
    tenant_id: str,
    original_file_name: str | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> int:
    """
    Background document ingestion task.

    The task runs the existing IngestionService outside
    the FastAPI request process and updates the document
    status in PostgreSQL.
    """

    if not document_id:
        raise ValueError(
            "document_id is required for document ingestion."
        )

    if not tenant_id:
        raise ValueError(
            "tenant_id is required for document ingestion."
        )

    document_uuid = UUID(document_id)
    tenant_uuid = UUID(tenant_id)

    try:
        ingestion_service = IngestionService()

        chunk_count = ingestion_service.ingest(
            storage_key=storage_key,
            tenant_id=tenant_uuid,
            original_file_name=original_file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        asyncio.run(
            _update_document_status(
                document_id=document_uuid,
                status="indexed",
            )
        )

        return chunk_count

    except Exception:
        try:
            asyncio.run(
                _update_document_status(
                    document_id=document_uuid,
                    status="failed",
                )
            )
        except Exception:
            # Preserve the original ingestion exception.
            pass

        raise