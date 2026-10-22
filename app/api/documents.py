from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import DocumentVersion
from app.ingestion.storage import save_upload
from app.rag.vectorstore.qdrant import QdrantService
from app.schemas.documents import DocumentResponse
from app.services.documents import (
    create_document,
    get_document_by_tenant,
    get_document_versions,
    get_documents_by_tenant,
)
from app.workers.tasks import ingest_document_task

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
}


MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2000
MAX_CHUNK_OVERLAP = 500


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def list_documents(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return documents belonging only to the current tenant.
    """

    documents = await get_documents_by_tenant(
        db=db,
        tenant_id=current_user["tenant_id"],
    )

    return documents


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(600),
    chunk_overlap: int = Form(150),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # -------------------------------------------------
    # FILE TYPE VALIDATION
    # -------------------------------------------------

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type",
        )

    # -------------------------------------------------
    # CHUNK PARAMETER VALIDATION
    # -------------------------------------------------

    if not (
        MIN_CHUNK_SIZE
        <= chunk_size
        <= MAX_CHUNK_SIZE
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"chunk_size must be between "
                f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}"
            ),
        )

    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "chunk_overlap must be smaller "
                "than chunk_size"
            ),
        )

    if not (
        0
        <= chunk_overlap
        <= MAX_CHUNK_OVERLAP
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"chunk_overlap must be between "
                f"0 and {MAX_CHUNK_OVERLAP}"
            ),
        )

    # -------------------------------------------------
    # SAVE UPLOAD
    # -------------------------------------------------

    storage_key, checksum = await save_upload(file)

    # -------------------------------------------------
    # CREATE DATABASE DOCUMENT
    # -------------------------------------------------

    document = await create_document(
        db=db,
        tenant_id=current_user["tenant_id"],
        name=file.filename or "unknown",
        mime_type=file.content_type,
        storage_key=storage_key,
        checksum=checksum,
    )

    # -------------------------------------------------
    # PROCESSING STATUS
    # -------------------------------------------------

    document.status = "processing"

    await db.commit()
    await db.refresh(document)

    # -------------------------------------------------
    # QUEUE BACKGROUND INGESTION
    # -------------------------------------------------

    try:
        ingest_document_task.delay(
            document_id=str(document.id),
            storage_key=storage_key,
            tenant_id=str(current_user["tenant_id"]),
            original_file_name=file.filename or "unknown",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    except Exception:
        document.status = "failed"

        await db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue document ingestion",
        )

    # -------------------------------------------------
    # RETURN IMMEDIATELY
    # -------------------------------------------------

    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document and its indexed chunks.

    Deletion is strictly tenant-scoped.
    """

    tenant_id = current_user["tenant_id"]

    # -------------------------------------------------
    # FIND DOCUMENT WITH TENANT ISOLATION
    # -------------------------------------------------

    document = await get_document_by_tenant(
        db=db,
        document_id=document_id,
        tenant_id=tenant_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # -------------------------------------------------
    # GET DOCUMENT VERSIONS
    # -------------------------------------------------

    versions = await get_document_versions(
        db=db,
        document_id=document.id,
    )

    qdrant_service = QdrantService()

    # -------------------------------------------------
    # DELETE QDRANT VECTORS
    # -------------------------------------------------

    for version in versions:
        try:
            qdrant_service.delete_documents_by_source(
                source=version.storage_key,
                tenant_id=tenant_id,
            )
        except Exception as exc:
            await db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to delete document vectors: "
                    f"{exc}"
                ),
            )

    # -------------------------------------------------
    # DELETE STORED FILES
    # -------------------------------------------------

    for version in versions:
        file_path = Path(version.storage_key)

        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as exc:
            await db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to delete stored document: "
                    f"{exc}"
                ),
            )

    # -------------------------------------------------
    # DELETE DOCUMENT VERSIONS
    # -------------------------------------------------

    await db.execute(
        delete(DocumentVersion).where(
            DocumentVersion.document_id == document.id
        )
    )

    # -------------------------------------------------
    # DELETE DOCUMENT
    # -------------------------------------------------

    await db.delete(document)

    await db.commit()

    return None