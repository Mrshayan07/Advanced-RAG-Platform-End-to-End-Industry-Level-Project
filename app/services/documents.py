from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentVersion


async def get_documents_by_tenant(
    db: AsyncSession,
    tenant_id: UUID,
):
    result = await db.execute(
        select(Document)
        .where(
            Document.tenant_id == tenant_id
        )
        .order_by(
            Document.created_at.desc()
        )
    )

    return result.scalars().all()


async def get_document_by_tenant(
    db: AsyncSession,
    document_id: UUID,
    tenant_id: UUID,
):
    result = await db.execute(
        select(Document)
        .where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
        )
    )

    return result.scalar_one_or_none()


async def get_document_versions(
    db: AsyncSession,
    document_id: UUID,
):
    result = await db.execute(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id == document_id
        )
        .order_by(
            DocumentVersion.version.desc()
        )
    )

    return result.scalars().all()


async def create_document(
    db: AsyncSession,
    tenant_id: UUID,
    name: str,
    mime_type: str,
    storage_key: str,
    checksum: str,
):
    document = Document(
        tenant_id=tenant_id,
        name=name,
        mime_type=mime_type,
        status="pending",
    )

    db.add(document)
    await db.flush()

    version = DocumentVersion(
        document_id=document.id,
        version=1,
        storage_key=storage_key,
        checksum=checksum,
    )

    db.add(version)

    await db.commit()
    await db.refresh(document)

    return document