
from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings

settings = get_settings()


class QdrantService:
    """
    Qdrant vector database service.

    Multi-tenant design:

        Every indexed chunk MUST contain:
            tenant_id

        Retrieval MUST provide:
            tenant_id

        Qdrant applies the tenant filter at the
        vector database level.

    Document deletion uses:

        tenant_id + source

    so that only chunks belonging to the requested
    tenant and document source are deleted.
    """

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

        self.collection_name = settings.qdrant_collection

    # ------------------------------------------------------------------
    # COLLECTION
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """
        Create the collection if it does not already exist.

        The collection uses 384-dimensional vectors because the
        current embedding model outputs 384-dimensional vectors.

        Payload indexes:

            tenant_id -> KEYWORD
            source    -> KEYWORD

        tenant_id is required for tenant-isolated retrieval.

        source is required for document-level vector deletion.
        """

        collections = self.client.get_collections()

        existing_collections = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

        # --------------------------------------------------------------
        # Tenant payload index
        # --------------------------------------------------------------
        #
        # tenant_id is stored as a string, therefore KEYWORD is
        # the correct payload index type.
        #
        # This index supports tenant-isolated retrieval.
        #
        # --------------------------------------------------------------

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

        except Exception as exc:
            error_message = str(exc).lower()

            # The index may already exist.
            if "already exists" not in error_message:
                raise

        # --------------------------------------------------------------
        # Source payload index
        # --------------------------------------------------------------
        #
        # Document deletion filters using:
        #
        #     tenant_id + source
        #
        # source is stored as a string, therefore KEYWORD is
        # the correct payload index type.
        #
        # --------------------------------------------------------------

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )

        except Exception as exc:
            error_message = str(exc).lower()

            # The index may already exist.
            if "already exists" not in error_message:
                raise

    # ------------------------------------------------------------------
    # POINT ID
    # ------------------------------------------------------------------

    def _generate_point_id(
        self,
        payload: dict[str, Any],
    ) -> str:
        """
        Generate a deterministic tenant-safe point ID.

        tenant_id is part of the ID so that:

            tenant-A + same source + same chunk

        and:

            tenant-B + same source + same chunk

        never collide.
        """

        tenant_id = payload.get("tenant_id")

        if not tenant_id:
            raise ValueError(
                "tenant_id is required in Qdrant payload."
            )

        source = payload.get("source", "")
        chunk_index = payload.get("chunk_index", 0)

        raw_id = (
            f"{tenant_id}:"
            f"{source}:"
            f"{chunk_index}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                raw_id,
            )
        )

    # ------------------------------------------------------------------
    # UPSERT VECTORS
    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Insert or update vectors and their payloads.

        Every payload MUST contain tenant_id.
        """

        if len(vectors) != len(payloads):
            raise ValueError(
                "Number of vectors must match number of payloads."
            )

        points: list[PointStruct] = []

        for vector, payload in zip(vectors, payloads):
            tenant_id = payload.get("tenant_id")

            if not tenant_id:
                raise ValueError(
                    "Every Qdrant payload must contain tenant_id."
                )

            point_id = self._generate_point_id(payload)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # ------------------------------------------------------------------
    # UPSERT DOCUMENTS
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        documents: list[Document],
        embedding_service,
        tenant_id: str | uuid.UUID,
    ) -> None:
        """
        Embed and index documents for a specific tenant.
        """

        normalized_tenant_id = str(tenant_id)

        if not normalized_tenant_id:
            raise ValueError(
                "tenant_id is required for document ingestion."
            )

        texts = [
            document.page_content
            for document in documents
        ]

        vectors = embedding_service.embed_documents(texts)

        payloads: list[dict[str, Any]] = []

        for document in documents:
            payload = {
                "text": document.page_content,
                **document.metadata,
                "tenant_id": normalized_tenant_id,
            }

            payloads.append(payload)

        self.upsert_vectors(
            vectors=vectors,
            payloads=payloads,
        )

    # ------------------------------------------------------------------
    # TENANT FILTER
    # ------------------------------------------------------------------

    @staticmethod
    def _tenant_filter(
        tenant_id: str | uuid.UUID,
    ) -> Filter:
        """
        Build a Qdrant filter that restricts results to one tenant.
        """

        normalized_tenant_id = str(tenant_id)

        if not normalized_tenant_id:
            raise ValueError(
                "tenant_id is required for Qdrant search."
            )

        return Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(
                        value=normalized_tenant_id,
                    ),
                )
            ]
        )

    # ------------------------------------------------------------------
    # VECTOR SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        tenant_id: str | uuid.UUID | None = None,
    ):
        """
        Perform tenant-isolated vector search.

        tenant_id is mandatory.

        The tenant filter is pushed down to Qdrant so documents
        belonging to other tenants never enter the retrieval result.
        """

        if tenant_id is None:
            raise ValueError(
                "tenant_id is required for Qdrant search."
            )

        start = time.perf_counter()

        tenant_filter = self._tenant_filter(
            tenant_id
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=tenant_filter,
            with_payload=True,
        )

        elapsed = time.perf_counter() - start

        print(
            "[Qdrant] Vector search: "
            f"{elapsed:.3f}s "
            f"(limit={limit}, "
            f"tenant_id={tenant_id})"
        )

        return results.points

    # ------------------------------------------------------------------
    # GET ALL DOCUMENTS
    # ------------------------------------------------------------------

    def get_all_documents(
        self,
        batch_size: int = 100,
        tenant_id: str | uuid.UUID | None = None,
    ) -> list[Document]:
        """
        Retrieve documents.

        If tenant_id is supplied, only that tenant's documents
        are returned.

        This is important for tenant-aware BM25 index construction.
        """

        tenant_filter = None

        if tenant_id is not None:
            tenant_filter = self._tenant_filter(
                tenant_id
            )

        documents: list[Document] = []

        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                scroll_filter=tenant_filter,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                text = payload.get(
                    "text",
                    "",
                )

                if not text:
                    continue

                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return documents

    # ------------------------------------------------------------------
    # DELETE DOCUMENT VECTORS
    # ------------------------------------------------------------------

    def delete_documents_by_source(
        self,
        source: str,
        tenant_id: str | uuid.UUID,
    ) -> None:
        """
        Delete all Qdrant chunks belonging to one source
        within one tenant.

        Tenant isolation is enforced at the Qdrant filter level.

        Filter:

            tenant_id + source
        """

        if not source:
            raise ValueError(
                "source is required for Qdrant document deletion."
            )

        tenant_filter = Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(
                        value=str(tenant_id),
                    ),
                ),
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=source,
                    ),
                ),
            ]
        )

        points_to_delete = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=tenant_filter,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            points_to_delete.extend(
                point.id for point in points
            )

            if next_offset is None:
                break

            offset = next_offset

        if not points_to_delete:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=points_to_delete,
        )
