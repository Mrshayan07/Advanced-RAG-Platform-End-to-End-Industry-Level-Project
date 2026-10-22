from __future__ import annotations

from langchain_core.documents import Document

from app.rag.retrieval.hybrid import HybridRetriever


TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


class FakeEmbeddingService:
    """
    Deterministic embedding service for isolation testing.

    We do not need real embedding models here because the
    purpose of this test is tenant propagation and isolation.
    """

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeQdrantService:
    """
    Fake Qdrant service.

    It simulates Qdrant's tenant filtering by returning
    documents only for the requested tenant.
    """

    def __init__(self):
        self.documents = {
            TENANT_A: [
                Document(
                    page_content=(
                        "Tenant A private document about "
                        "artificial intelligence."
                    ),
                    metadata={
                        "file_name": "tenant_a_ai.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_A,
                    },
                ),
                Document(
                    page_content=(
                        "Tenant A private document about "
                        "machine learning."
                    ),
                    metadata={
                        "file_name": "tenant_a_ml.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_A,
                    },
                ),
            ],
            TENANT_B: [
                Document(
                    page_content=(
                        "Tenant B private document about "
                        "artificial intelligence."
                    ),
                    metadata={
                        "file_name": "tenant_b_ai.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_B,
                    },
                ),
                Document(
                    page_content=(
                        "Tenant B private document about "
                        "machine learning."
                    ),
                    metadata={
                        "file_name": "tenant_b_ml.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_B,
                    },
                ),
            ],
        }

        self.requested_tenants: list[str] = []

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        tenant_id: str | None = None,
    ):
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for fake Qdrant search."
            )

        self.requested_tenants.append(
            str(tenant_id)
        )

        return [
            type(
                "FakeQdrantResult",
                (),
                {
                    "payload": {
                        "text": document.page_content,
                        **document.metadata,
                    }
                },
            )()
            for document in self.documents.get(
                str(tenant_id),
                [],
            )[:limit]
        ]


class FakeBM25Service:
    """
    Fake tenant-aware BM25 service.

    This verifies that HybridRetriever actually forwards
    tenant_id into BM25Service.search().
    """

    def __init__(self):
        self.documents = {
            TENANT_A: [
                Document(
                    page_content=(
                        "Tenant A private document about "
                        "artificial intelligence."
                    ),
                    metadata={
                        "file_name": "tenant_a_ai.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_A,
                    },
                ),
                Document(
                    page_content=(
                        "Tenant A private document about "
                        "machine learning."
                    ),
                    metadata={
                        "file_name": "tenant_a_ml.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_A,
                    },
                ),
            ],
            TENANT_B: [
                Document(
                    page_content=(
                        "Tenant B private document about "
                        "artificial intelligence."
                    ),
                    metadata={
                        "file_name": "tenant_b_ai.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_B,
                    },
                ),
                Document(
                    page_content=(
                        "Tenant B private document about "
                        "machine learning."
                    ),
                    metadata={
                        "file_name": "tenant_b_ml.pdf",
                        "chunk_index": 0,
                        "tenant_id": TENANT_B,
                    },
                ),
            ],
        }

        self.requested_tenants: list[str] = []

    def search(
        self,
        query: str,
        tenant_id: str,
        k: int = 5,
    ) -> list[Document]:
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for fake BM25 search."
            )

        self.requested_tenants.append(
            str(tenant_id)
        )

        return self.documents.get(
            str(tenant_id),
            [],
        )[:k]


class FakeQueryRewriter:
    """
    Deterministic query rewriter.

    No LLM call is needed for this isolation test.
    """

    def rewrite(
        self,
        question: str,
        chat_history: str = "",
    ) -> str:
        return question


class FakeReranker:
    """
    Deterministic reranker.

    It keeps the received documents in the same order and
    adds a predictable rerank score.

    This verifies that tenant validation continues through
    the reranking stage.
    """

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        results = documents[:top_k]

        for index, document in enumerate(
            results,
            start=1,
        ):
            document.metadata["rerank_score"] = (
                1.0 / index
            )

        return results


def assert_tenant_isolation(
    documents: list[Document],
    expected_tenant: str,
    stage: str,
) -> None:
    """
    Assert that every returned document belongs to the
    expected tenant.
    """

    for document in documents:
        actual_tenant = document.metadata.get(
            "tenant_id"
        )

        assert str(actual_tenant) == str(
            expected_tenant
        ), (
            f"Tenant isolation violation at {stage}: "
            f"expected={expected_tenant}, "
            f"got={actual_tenant}"
        )


def build_test_retriever():
    """
    Build HybridRetriever with deterministic test doubles.
    """

    bm25_service = FakeBM25Service()
    qdrant_service = FakeQdrantService()
    embedding_service = FakeEmbeddingService()
    query_rewriter = FakeQueryRewriter()
    reranker = FakeReranker()

    retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
        enable_logging=False,
    )

    return (
        retriever,
        bm25_service,
        qdrant_service,
    )


def test_tenant_a_hybrid_retrieval():
    """
    Tenant A must receive only Tenant A documents.
    """

    (
        retriever,
        bm25_service,
        qdrant_service,
    ) = build_test_retriever()

    results = retriever.search(
        query="artificial intelligence",
        rewritten_query="artificial intelligence",
        dense_k=10,
        bm25_k=10,
        top_k=5,
        tenant_id=TENANT_A,
    )

    assert results, (
        "Tenant A should receive retrieval results."
    )

    assert_tenant_isolation(
        documents=results,
        expected_tenant=TENANT_A,
        stage="Tenant A final hybrid retrieval",
    )

    assert TENANT_A in (
        qdrant_service.requested_tenants
    ), (
        "HybridRetriever did not pass Tenant A "
        "to Qdrant."
    )

    assert TENANT_A in (
        bm25_service.requested_tenants
    ), (
        "HybridRetriever did not pass Tenant A "
        "to BM25."
    )

    print(
        "[PASS] Tenant A hybrid retrieval "
        "contains only Tenant A documents."
    )


def test_tenant_b_hybrid_retrieval():
    """
    Tenant B must receive only Tenant B documents.
    """

    (
        retriever,
        bm25_service,
        qdrant_service,
    ) = build_test_retriever()

    results = retriever.search(
        query="artificial intelligence",
        rewritten_query="artificial intelligence",
        dense_k=10,
        bm25_k=10,
        top_k=5,
        tenant_id=TENANT_B,
    )

    assert results, (
        "Tenant B should receive retrieval results."
    )

    assert_tenant_isolation(
        documents=results,
        expected_tenant=TENANT_B,
        stage="Tenant B final hybrid retrieval",
    )

    assert TENANT_B in (
        qdrant_service.requested_tenants
    ), (
        "HybridRetriever did not pass Tenant B "
        "to Qdrant."
    )

    assert TENANT_B in (
        bm25_service.requested_tenants
    ), (
        "HybridRetriever did not pass Tenant B "
        "to BM25."
    )

    print(
        "[PASS] Tenant B hybrid retrieval "
        "contains only Tenant B documents."
    )


def test_bidirectional_hybrid_isolation():
    """
    Verify that retrieving for A and B independently
    never causes documents from the other tenant to appear.
    """

    (
        retriever,
        _,
        _,
    ) = build_test_retriever()

    tenant_a_results = retriever.search(
        query="machine learning",
        rewritten_query="machine learning",
        dense_k=10,
        bm25_k=10,
        top_k=5,
        tenant_id=TENANT_A,
    )

    tenant_b_results = retriever.search(
        query="machine learning",
        rewritten_query="machine learning",
        dense_k=10,
        bm25_k=10,
        top_k=5,
        tenant_id=TENANT_B,
    )

    assert_tenant_isolation(
        documents=tenant_a_results,
        expected_tenant=TENANT_A,
        stage="Bidirectional Tenant A retrieval",
    )

    assert_tenant_isolation(
        documents=tenant_b_results,
        expected_tenant=TENANT_B,
        stage="Bidirectional Tenant B retrieval",
    )

    tenant_a_files = {
        document.metadata.get("file_name")
        for document in tenant_a_results
    }

    tenant_b_files = {
        document.metadata.get("file_name")
        for document in tenant_b_results
    }

    assert tenant_a_files.isdisjoint(
        tenant_b_files
    ), (
        "Cross-tenant document overlap detected "
        "between Tenant A and Tenant B."
    )

    print(
        "[PASS] Bidirectional hybrid tenant isolation "
        "verified."
    )


def test_unknown_tenant_returns_no_documents():
    """
    An unknown tenant must not receive documents belonging
    to Tenant A or Tenant B.
    """

    (
        retriever,
        _,
        _,
    ) = build_test_retriever()

    unknown_tenant = "tenant-unknown"

    results = retriever.search(
        query="artificial intelligence",
        rewritten_query="artificial intelligence",
        dense_k=10,
        bm25_k=10,
        top_k=5,
        tenant_id=unknown_tenant,
    )

    assert results == [], (
        "Unknown tenant must not receive documents "
        "from another tenant."
    )

    print(
        "[PASS] Unknown tenant receives zero documents."
    )


def test_missing_tenant_id_is_rejected():
    """
    tenant_id is mandatory for HybridRetriever.
    """

    (
        retriever,
        _,
        _,
    ) = build_test_retriever()

    try:
        retriever.search(
            query="artificial intelligence",
            rewritten_query="artificial intelligence",
            tenant_id=None,
        )

    except ValueError as exc:
        assert (
            "tenant_id is required"
            in str(exc)
        )

        print(
            "[PASS] Missing tenant_id correctly rejected."
        )

        return

    raise AssertionError(
        "HybridRetriever accepted missing tenant_id."
    )


def test_search_with_scores_is_tenant_safe():
    """
    search_with_scores() must also preserve tenant isolation.
    """

    (
        retriever,
        _,
        _,
    ) = build_test_retriever()

    results = retriever.search_with_scores(
        query="artificial intelligence",
        top_k=5,
        tenant_id=TENANT_A,
    )

    assert results, (
        "Tenant A should receive scored results."
    )

    documents = [
        document
        for document, score in results
    ]

    assert_tenant_isolation(
        documents=documents,
        expected_tenant=TENANT_A,
        stage="search_with_scores",
    )

    for document, score in results:
        assert isinstance(
            score,
            float,
        ), (
            "Reranker score must be a float."
        )

    print(
        "[PASS] search_with_scores() preserves "
        "tenant isolation."
    )


def main():
    print("\n" + "=" * 70)
    print("HYBRID RETRIEVER TENANT ISOLATION TEST")
    print("=" * 70)

    test_tenant_a_hybrid_retrieval()
    test_tenant_b_hybrid_retrieval()
    test_bidirectional_hybrid_isolation()
    test_unknown_tenant_returns_no_documents()
    test_missing_tenant_id_is_rejected()
    test_search_with_scores_is_tenant_safe()

    print("\n" + "=" * 70)
    print(
        "ALL HYBRID TENANT ISOLATION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()