from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.loaders.text import load_text_file
from app.ingestion.processors.metadata import enrich_metadata
from app.ingestion.processors.splitter import split_documents
from app.rag.embeddings.service import EmbeddingService
from app.rag.retrieval.bm25 import BM25Service
from app.rag.vectorstore.qdrant import QdrantService


class IngestionService:
    """
    Loads, chunks, enriches, embeds, and indexes documents.

    Tenant-aware ingestion flow:

        File
          ↓
        Loader
          ↓
        Dynamic Chunking
          ↓
        Metadata enrichment
          ↓
        Qdrant tenant-scoped upsert
          ↓
        Reload tenant documents from Qdrant
          ↓
        Refresh tenant-specific BM25 index

    Every document indexed into Qdrant belongs to the
    authenticated tenant.

    After successful ingestion, the tenant's BM25 index
    is refreshed so that the newly uploaded document becomes
    immediately searchable through hybrid retrieval.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.bm25_service = BM25Service()

    def ingest(
        self,
        storage_key: str,
        tenant_id: UUID | str,
        original_file_name: str | None = None,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
    ) -> int:
        """
        Ingest a document for a specific tenant.

        Args:
            storage_key:
                Path to the stored document.

            tenant_id:
                Authenticated tenant identifier.

            chunk_size:
                Maximum character size used when splitting
                oversized logical sections.

            chunk_overlap:
                Number of overlapping characters between
                consecutive chunks.

        Returns:
            int: Number of chunks indexed.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for document ingestion."
            )

        normalized_tenant_id = str(tenant_id)

        file_path = Path(storage_key)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document file not found: {file_path}"
            )

        # -------------------------------------------------
        # LOAD DOCUMENT
        # -------------------------------------------------

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            documents = load_pdf_file(
                str(file_path)
            )

        elif suffix in {".txt", ".text"}:
            documents = load_text_file(
                str(file_path)
            )

        else:
            raise ValueError(
                f"Unsupported document type: {suffix}"
            )

        if not documents:
            raise ValueError(
                f"No content extracted from: {file_path}"
            )

        # -------------------------------------------------
        # DYNAMIC CHUNKING
        # -------------------------------------------------

        chunks = split_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise ValueError(
                f"No chunks generated from: {file_path}"
            )

        # -------------------------------------------------
        # ENRICH METADATA
        # -------------------------------------------------

        chunks = enrich_metadata(
            chunks,
            original_file_name=original_file_name,
        )

        # -------------------------------------------------
        # QDRANT
        # -------------------------------------------------

        self.qdrant_service.create_collection()

        self.qdrant_service.upsert_documents(
            documents=chunks,
            embedding_service=self.embedding_service,
            tenant_id=tenant_id,
        )

        # -------------------------------------------------
        # BM25 REFRESH
        # -------------------------------------------------

        self._refresh_tenant_bm25(
            tenant_id=normalized_tenant_id
        )

        return len(chunks)

    def _refresh_tenant_bm25(
        self,
        tenant_id: str,
    ) -> None:
        """
        Rebuild the BM25 index using all currently indexed
        documents belonging to one tenant.

        Important:

            We DO NOT rebuild all tenants.

        Only:

            tenant_id
                ↓
            Qdrant tenant filter
                ↓
            tenant documents
                ↓
            BM25 tenant index
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for BM25 refresh."
            )

        tenant_documents = (
            self.qdrant_service.get_all_documents(
                tenant_id=tenant_id
            )
        )

        if not tenant_documents:
            self.bm25_service.remove_tenant(
                tenant_id
            )

            print(
                "[Ingestion] BM25 index removed because "
                f"tenant={tenant_id} has no indexed documents."
            )

            return

        self.bm25_service.refresh_index(
            documents=tenant_documents,
            tenant_id=tenant_id,
        )

        print(
            "[Ingestion] BM25 index refreshed: "
            f"tenant={tenant_id}, "
            f"documents={len(tenant_documents)}"
        )