
from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from app.rag.retrieval.bm25 import BM25Service
from app.services.ingestion import IngestionService


def main() -> None:
    print("=" * 70)
    print("INGESTION → QDRANT → BM25 REFRESH INTEGRATION TEST")
    print("=" * 70)

    tenant_a = str(uuid4())
    tenant_b = str(uuid4())

    print(f"\nTenant A: {tenant_a}")
    print(f"Tenant B: {tenant_b}")

    # ---------------------------------------------------------
    # CLEAN BM25 STATE
    # ---------------------------------------------------------

    bm25 = BM25Service()
    bm25.clear()

    # ---------------------------------------------------------
    # CREATE TEMPORARY DOCUMENTS
    # ---------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        tenant_a_file = temp_path / "tenant_a_rag.txt"
        tenant_b_file = temp_path / "tenant_b_python.txt"

        tenant_a_file.write_text(
            (
                "Tenant A private knowledge about "
                "retrieval augmented generation RAG, "
                "vector databases, embeddings, and "
                "large language models."
            ),
            encoding="utf-8",
        )

        tenant_b_file.write_text(
            (
                "Tenant B private knowledge about "
                "Python programming, FastAPI, "
                "and backend development."
            ),
            encoding="utf-8",
        )

        # -----------------------------------------------------
        # CREATE INGESTION SERVICES
        # -----------------------------------------------------

        ingestion_a = IngestionService()
        ingestion_b = IngestionService()

        # -----------------------------------------------------
        # VERIFY BOTH SERVICES SHARE BM25 INSTANCE
        # -----------------------------------------------------

        assert (
            ingestion_a.bm25_service
            is ingestion_b.bm25_service
        )

        print(
            "[PASS] IngestionService instances share "
            "the same BM25Service singleton."
        )

        # -----------------------------------------------------
        # INGEST TENANT A DOCUMENT
        # -----------------------------------------------------

        print(
            "\n[INFO] Ingesting Tenant A document..."
        )

        chunks_a = ingestion_a.ingest(
            storage_key=str(tenant_a_file),
            tenant_id=tenant_a,
        )

        assert chunks_a > 0

        print(
            f"[PASS] Tenant A document ingested "
            f"with {chunks_a} chunk(s)."
        )

        # -----------------------------------------------------
        # VERIFY TENANT A BM25 INDEX
        # -----------------------------------------------------

        assert bm25.has_index(tenant_a)

        print(
            "[PASS] Tenant A BM25 index was automatically "
            "created/refreshed by ingestion."
        )

        # -----------------------------------------------------
        # SEARCH TENANT A
        # -----------------------------------------------------

        results_a = bm25.search(
            query="retrieval augmented generation",
            tenant_id=tenant_a,
            k=5,
        )

        assert results_a

        for document in results_a:
            assert str(
                document.metadata.get("tenant_id")
            ) == tenant_a

        print(
            "[PASS] Tenant A document is searchable "
            "through BM25 after ingestion."
        )

        # -----------------------------------------------------
        # INGEST TENANT B DOCUMENT
        # -----------------------------------------------------

        print(
            "\n[INFO] Ingesting Tenant B document..."
        )

        chunks_b = ingestion_b.ingest(
            storage_key=str(tenant_b_file),
            tenant_id=tenant_b,
        )

        assert chunks_b > 0

        print(
            f"[PASS] Tenant B document ingested "
            f"with {chunks_b} chunk(s)."
        )

        # -----------------------------------------------------
        # VERIFY TENANT B BM25 INDEX
        # -----------------------------------------------------

        assert bm25.has_index(tenant_b)

        print(
            "[PASS] Tenant B BM25 index was automatically "
            "created/refreshed by ingestion."
        )

        # -----------------------------------------------------
        # SEARCH TENANT B
        # -----------------------------------------------------

        results_b = bm25.search(
            query="Python FastAPI backend",
            tenant_id=tenant_b,
            k=5,
        )

        assert results_b

        for document in results_b:
            assert str(
                document.metadata.get("tenant_id")
            ) == tenant_b

        print(
            "[PASS] Tenant B document is searchable "
            "through BM25 after ingestion."
        )

        # -----------------------------------------------------
        # CROSS-TENANT ISOLATION
        # -----------------------------------------------------

        cross_a = bm25.search(
            query="Python FastAPI backend",
            tenant_id=tenant_a,
            k=5,
        )

        for document in cross_a:
            assert str(
                document.metadata.get("tenant_id")
            ) == tenant_a

        assert all(
            "Python programming"
            not in document.page_content
            for document in cross_a
        )

        print(
            "[PASS] Tenant A cannot retrieve "
            "Tenant B private content."
        )

        cross_b = bm25.search(
            query="retrieval augmented generation",
            tenant_id=tenant_b,
            k=5,
        )

        for document in cross_b:
            assert str(
                document.metadata.get("tenant_id")
            ) == tenant_b

        assert all(
            "retrieval augmented generation"
            not in document.page_content
            for document in cross_b
        )

        print(
            "[PASS] Tenant B cannot retrieve "
            "Tenant A private content."
        )

        # -----------------------------------------------------
        # VERIFY TENANT INDEX COUNT
        # -----------------------------------------------------

        assert bm25.tenant_count() == 2

        print(
            "[PASS] BM25 contains exactly two tenant indexes."
        )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    bm25.clear()

    assert bm25.tenant_count() == 0

    print(
        "[PASS] BM25 test indexes cleaned up."
    )

    print("\n" + "=" * 70)
    print(
        "ALL INGESTION → QDRANT → BM25 INTEGRATION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
