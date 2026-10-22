
from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25Service:
    """
    Tenant-aware BM25 retrieval service.

    Architecture:

        tenant_id
            ↓
        BM25Retriever for that tenant
            ↓
        Tenant-specific documents

    Each tenant gets an independent BM25 index.

    The service is implemented as a process-local singleton so that:

        RAG startup
            ↓
        BM25Service()

        and

        Document ingestion
            ↓
        BM25Service()

    reference the same in-memory BM25 indexes.

    This allows newly ingested documents to refresh the
    BM25 index used by the active RAG pipeline.
    """

    _instance: BM25Service | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.retrievers: dict[str, BM25Retriever] = {}

        self._initialized = True

    def build_index(
        self,
        documents: list[Document],
        tenant_id: str,
    ) -> None:
        """
        Build or replace the BM25 index for one tenant.

        Existing index for the same tenant is replaced.

        This method never mixes documents from different tenants.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required to build BM25 index."
            )

        if not documents:
            raise ValueError(
                "Cannot build BM25 index with no documents."
            )

        normalized_tenant_id = str(tenant_id)

        for document in documents:
            document_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if document_tenant_id is None:
                raise ValueError(
                    "Every BM25 document must contain tenant_id."
                )

            if str(document_tenant_id) != normalized_tenant_id:
                raise ValueError(
                    "BM25 document tenant_id does not match "
                    f"requested tenant_id={normalized_tenant_id}."
                )

        retriever = BM25Retriever.from_documents(
            documents
        )

        self.retrievers[normalized_tenant_id] = retriever

    def refresh_index(
        self,
        documents: list[Document],
        tenant_id: str,
    ) -> None:
        """
        Refresh the BM25 index for a tenant.

        This is intentionally an alias around build_index().

        The important semantic difference is that callers can
        explicitly express their intention:

            document ingestion
                ↓
            refresh tenant BM25 index

        The complete tenant document set should be supplied.
        """

        self.build_index(
            documents=documents,
            tenant_id=tenant_id,
        )

    def search(
        self,
        query: str,
        tenant_id: str,
        k: int = 5,
    ) -> list[Document]:
        """
        Search only the BM25 index belonging to tenant_id.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for BM25 search."
            )

        normalized_tenant_id = str(tenant_id)

        retriever = self.retrievers.get(
            normalized_tenant_id
        )

        if retriever is None:
            raise RuntimeError(
                "BM25 index is not initialized for "
                f"tenant_id={normalized_tenant_id}. "
                "Call build_index() first."
            )

        retriever.k = k

        results = retriever.invoke(query)

        for document in results:
            result_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if str(result_tenant_id) != normalized_tenant_id:
                raise RuntimeError(
                    "BM25 tenant isolation violation detected. "
                    f"Expected tenant_id={normalized_tenant_id}, "
                    f"got tenant_id={result_tenant_id}."
                )

        return results

    def remove_tenant(
        self,
        tenant_id: str,
    ) -> None:
        """
        Remove one tenant's BM25 index.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required to remove BM25 index."
            )

        self.retrievers.pop(
            str(tenant_id),
            None,
        )

    def clear(self) -> None:
        """
        Remove all in-memory tenant BM25 indexes.
        """

        self.retrievers.clear()

    def has_index(
        self,
        tenant_id: str,
    ) -> bool:
        """
        Check whether a tenant has an initialized BM25 index.
        """

        if not tenant_id:
            return False

        return str(tenant_id) in self.retrievers

    def tenant_count(self) -> int:
        """
        Return the number of tenant BM25 indexes currently loaded.
        """

        return len(self.retrievers)
