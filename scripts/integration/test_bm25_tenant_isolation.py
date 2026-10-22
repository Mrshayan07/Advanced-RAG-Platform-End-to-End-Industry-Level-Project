from __future__ import annotations

import uuid

from langchain_core.documents import Document

from app.rag.retrieval.bm25 import BM25Service


def main() -> None:
    print("=" * 70)
    print("BM25 MULTI-TENANT ISOLATION TEST")
    print("=" * 70)

    tenant_a = str(uuid.uuid4())
    tenant_b = str(uuid.uuid4())
    unknown_tenant = str(uuid.uuid4())

    print(f"\nTenant A: {tenant_a}")
    print(f"Tenant B: {tenant_b}")

    bm25 = BM25Service()

    # ---------------------------------------------------------
    # Tenant A documents
    # ---------------------------------------------------------

    tenant_a_documents = [
        Document(
            page_content=(
                "Tenant A secret architecture document "
                "about artificial intelligence systems."
            ),
            metadata={
                "tenant_id": tenant_a,
                "source": "shared.txt",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "Tenant A private information about "
                "machine learning models."
            ),
            metadata={
                "tenant_id": tenant_a,
                "source": "private-a.txt",
                "chunk_index": 1,
            },
        ),
    ]

    # ---------------------------------------------------------
    # Tenant B documents
    # ---------------------------------------------------------

    tenant_b_documents = [
        Document(
            page_content=(
                "Tenant B secret architecture document "
                "about artificial intelligence systems."
            ),
            metadata={
                "tenant_id": tenant_b,
                "source": "shared.txt",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "Tenant B private information about "
                "machine learning models."
            ),
            metadata={
                "tenant_id": tenant_b,
                "source": "private-b.txt",
                "chunk_index": 1,
            },
        ),
    ]

    # ---------------------------------------------------------
    # 1. Build Tenant A index
    # ---------------------------------------------------------

    bm25.build_index(
        documents=tenant_a_documents,
        tenant_id=tenant_a,
    )

    assert bm25.has_index(tenant_a)

    print(
        "[PASS] Tenant A BM25 index created."
    )

    # ---------------------------------------------------------
    # 2. Build Tenant B index
    # ---------------------------------------------------------

    bm25.build_index(
        documents=tenant_b_documents,
        tenant_id=tenant_b,
    )

    assert bm25.has_index(tenant_b)

    print(
        "[PASS] Tenant B BM25 index created."
    )

    # ---------------------------------------------------------
    # 3. Verify two independent indexes exist
    # ---------------------------------------------------------

    assert bm25.tenant_count() == 2

    print(
        "[PASS] Separate BM25 indexes exist for both tenants."
    )

    # ---------------------------------------------------------
    # 4. Tenant A search
    # ---------------------------------------------------------

    tenant_a_results = bm25.search(
        query="artificial intelligence architecture",
        tenant_id=tenant_a,
        k=5,
    )

    assert tenant_a_results

    tenant_a_result_ids = {
        document.metadata.get("tenant_id")
        for document in tenant_a_results
    }

    assert tenant_a_result_ids == {tenant_a}

    tenant_a_texts = {
        document.page_content
        for document in tenant_a_results
    }

    assert any(
        "Tenant A" in text
        for text in tenant_a_texts
    )

    assert not any(
        "Tenant B" in text
        for text in tenant_a_texts
    )

    print(
        "[PASS] Tenant A retrieves only Tenant A documents."
    )

    # ---------------------------------------------------------
    # 5. Tenant B search
    # ---------------------------------------------------------

    tenant_b_results = bm25.search(
        query="artificial intelligence architecture",
        tenant_id=tenant_b,
        k=5,
    )

    assert tenant_b_results

    tenant_b_result_ids = {
        document.metadata.get("tenant_id")
        for document in tenant_b_results
    }

    assert tenant_b_result_ids == {tenant_b}

    tenant_b_texts = {
        document.page_content
        for document in tenant_b_results
    }

    assert any(
        "Tenant B" in text
        for text in tenant_b_texts
    )

    assert not any(
        "Tenant A" in text
        for text in tenant_b_texts
    )

    print(
        "[PASS] Tenant B retrieves only Tenant B documents."
    )

    # ---------------------------------------------------------
    # 6. Unknown tenant
    # ---------------------------------------------------------

    try:
        bm25.search(
            query="artificial intelligence",
            tenant_id=unknown_tenant,
            k=5,
        )

        raise AssertionError(
            "Unknown tenant should not have a BM25 index."
        )

    except RuntimeError as exc:
        assert "not initialized" in str(exc).lower()

    print(
        "[PASS] Unknown tenant cannot access another "
        "tenant's BM25 index."
    )

    # ---------------------------------------------------------
    # 7. Missing tenant_id during index creation
    # ---------------------------------------------------------

    try:
        bm25.build_index(
            documents=tenant_a_documents,
            tenant_id="",
        )

        raise AssertionError(
            "Missing tenant_id should be rejected."
        )

    except ValueError as exc:
        assert "tenant_id" in str(exc).lower()

    print(
        "[PASS] Missing tenant_id rejected during index creation."
    )

    # ---------------------------------------------------------
    # 8. Mixed-tenant documents must be rejected
    # ---------------------------------------------------------

    mixed_documents = [
        tenant_a_documents[0],
        tenant_b_documents[0],
    ]

    try:
        bm25.build_index(
            documents=mixed_documents,
            tenant_id=tenant_a,
        )

        raise AssertionError(
            "Mixed-tenant BM25 documents should be rejected."
        )

    except ValueError as exc:
        assert "tenant_id" in str(exc).lower()

    print(
        "[PASS] Mixed-tenant documents rejected."
    )

    # ---------------------------------------------------------
    # 9. Remove Tenant A index
    # ---------------------------------------------------------

    bm25.remove_tenant(tenant_a)

    assert not bm25.has_index(tenant_a)
    assert bm25.has_index(tenant_b)

    print(
        "[PASS] Tenant A index removed without affecting Tenant B."
    )

    # ---------------------------------------------------------
    # 10. Tenant B still works
    # ---------------------------------------------------------

    tenant_b_results_after_removal = bm25.search(
        query="machine learning models",
        tenant_id=tenant_b,
        k=5,
    )

    assert tenant_b_results_after_removal

    assert all(
        document.metadata.get("tenant_id") == tenant_b
        for document in tenant_b_results_after_removal
    )

    print(
        "[PASS] Tenant B index remains functional."
    )

    # ---------------------------------------------------------
    # 11. Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL BM25 TENANT ISOLATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()