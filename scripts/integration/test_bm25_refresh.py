
from __future__ import annotations

from uuid import uuid4

from langchain_core.documents import Document

from app.rag.retrieval.bm25 import BM25Service


def make_document(
    tenant_id: str,
    file_name: str,
    chunk_index: int,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "tenant_id": tenant_id,
            "file_name": file_name,
            "chunk_index": chunk_index,
        },
    )


def assert_only_tenant(
    documents: list[Document],
    expected_tenant_id: str,
) -> None:
    for document in documents:
        actual_tenant_id = str(
            document.metadata.get("tenant_id")
        )

        assert actual_tenant_id == expected_tenant_id, (
            "Tenant isolation violation: "
            f"expected={expected_tenant_id}, "
            f"got={actual_tenant_id}"
        )


def main() -> None:
    print("=" * 70)
    print("BM25 INDEX REFRESH + TENANT ISOLATION TEST")
    print("=" * 70)

    tenant_a = str(uuid4())
    tenant_b = str(uuid4())

    print(f"\nTenant A: {tenant_a}")
    print(f"Tenant B: {tenant_b}")

    bm25 = BM25Service()

    # ---------------------------------------------------------
    # CLEAN START
    # ---------------------------------------------------------

    bm25.clear()

    # ---------------------------------------------------------
    # INITIAL TENANT A DOCUMENTS
    # ---------------------------------------------------------

    tenant_a_initial = [
        make_document(
            tenant_id=tenant_a,
            file_name="tenant_a_ai.pdf",
            chunk_index=0,
            content=(
                "Tenant A document about artificial intelligence "
                "and neural networks."
            ),
        ),
        make_document(
            tenant_id=tenant_a,
            file_name="tenant_a_python.pdf",
            chunk_index=0,
            content=(
                "Tenant A document about Python programming "
                "and software development."
            ),
        ),
    ]

    bm25.build_index(
        documents=tenant_a_initial,
        tenant_id=tenant_a,
    )

    assert bm25.has_index(tenant_a)
    print("[PASS] Tenant A initial BM25 index created.")

    # ---------------------------------------------------------
    # INITIAL TENANT B DOCUMENTS
    # ---------------------------------------------------------

    tenant_b_initial = [
        make_document(
            tenant_id=tenant_b,
            file_name="tenant_b_database.pdf",
            chunk_index=0,
            content=(
                "Tenant B document about PostgreSQL databases "
                "and SQL queries."
            ),
        ),
        make_document(
            tenant_id=tenant_b,
            file_name="tenant_b_cloud.pdf",
            chunk_index=0,
            content=(
                "Tenant B document about cloud computing "
                "and AWS infrastructure."
            ),
        ),
    ]

    bm25.build_index(
        documents=tenant_b_initial,
        tenant_id=tenant_b,
    )

    assert bm25.has_index(tenant_b)
    print("[PASS] Tenant B initial BM25 index created.")

    assert bm25.tenant_count() == 2
    print("[PASS] Two independent tenant BM25 indexes exist.")

    # ---------------------------------------------------------
    # VERIFY INITIAL TENANT A SEARCH
    # ---------------------------------------------------------

    results_a_before = bm25.search(
        query="artificial intelligence",
        tenant_id=tenant_a,
        k=5,
    )

    assert results_a_before
    assert_only_tenant(
        results_a_before,
        tenant_a,
    )

    print(
        "[PASS] Tenant A initial search returns only "
        "Tenant A documents."
    )

    # ---------------------------------------------------------
    # VERIFY INITIAL TENANT B SEARCH
    # ---------------------------------------------------------

    results_b_before = bm25.search(
        query="cloud computing",
        tenant_id=tenant_b,
        k=5,
    )

    assert results_b_before
    assert_only_tenant(
        results_b_before,
        tenant_b,
    )

    print(
        "[PASS] Tenant B initial search returns only "
        "Tenant B documents."
    )

    # ---------------------------------------------------------
    # ADD NEW DOCUMENT TO TENANT A
    # ---------------------------------------------------------

    tenant_a_new_document = make_document(
        tenant_id=tenant_a,
        file_name="tenant_a_llm.pdf",
        chunk_index=0,
        content=(
            "Tenant A new document about large language models "
            "LLMs retrieval augmented generation and RAG."
        ),
    )

    tenant_a_after_upload = [
        *tenant_a_initial,
        tenant_a_new_document,
    ]

    print(
        "\n[INFO] Adding a new document to Tenant A..."
    )

    # ---------------------------------------------------------
    # REFRESH ONLY TENANT A
    # ---------------------------------------------------------

    bm25.refresh_index(
        documents=tenant_a_after_upload,
        tenant_id=tenant_a,
    )

    print(
        "[PASS] Tenant A BM25 index refreshed."
    )

    # ---------------------------------------------------------
    # VERIFY NEW TENANT A DOCUMENT IS SEARCHABLE
    # ---------------------------------------------------------

    results_a_after = bm25.search(
        query="large language models RAG",
        tenant_id=tenant_a,
        k=5,
    )

    assert results_a_after
    assert_only_tenant(
        results_a_after,
        tenant_a,
    )

    new_document_found = any(
        document.metadata.get("file_name")
        == "tenant_a_llm.pdf"
        for document in results_a_after
    )

    assert new_document_found, (
        "New Tenant A document was not found "
        "after BM25 refresh."
    )

    print(
        "[PASS] New Tenant A document is searchable "
        "after BM25 refresh."
    )

    # ---------------------------------------------------------
    # VERIFY TENANT A OLD DOCUMENTS STILL EXIST
    # ---------------------------------------------------------

    all_a_results = bm25.search(
        query="Python artificial intelligence",
        tenant_id=tenant_a,
        k=5,
    )

    assert_only_tenant(
        all_a_results,
        tenant_a,
    )

    file_names_a = {
        document.metadata.get("file_name")
        for document in all_a_results
    }

    assert "tenant_a_ai.pdf" in file_names_a
    assert "tenant_a_python.pdf" in file_names_a

    print(
        "[PASS] Tenant A existing documents remain "
        "after refresh."
    )

    # ---------------------------------------------------------
    # VERIFY TENANT B WAS NOT AFFECTED
    # ---------------------------------------------------------

    results_b_after = bm25.search(
        query="cloud computing",
        tenant_id=tenant_b,
        k=5,
    )

    assert results_b_after
    assert_only_tenant(
        results_b_after,
        tenant_b,
    )

    file_names_b = {
        document.metadata.get("file_name")
        for document in results_b_after
    }

    assert "tenant_b_cloud.pdf" in file_names_b
    assert "tenant_b_database.pdf" in file_names_b

    assert "tenant_a_llm.pdf" not in file_names_b

    print(
        "[PASS] Tenant B BM25 index remains unchanged "
        "after Tenant A refresh."
    )

    # ---------------------------------------------------------
    # VERIFY TENANT B CANNOT SEE TENANT A NEW DOCUMENT
    # ---------------------------------------------------------

    cross_tenant_results = bm25.search(
        query="large language models RAG",
        tenant_id=tenant_b,
        k=5,
    )

    assert_only_tenant(
        cross_tenant_results,
        tenant_b,
    )

    assert all(
        document.metadata.get("file_name")
        != "tenant_a_llm.pdf"
        for document in cross_tenant_results
    )

    print(
        "[PASS] Tenant B cannot retrieve Tenant A's "
        "new document."
    )

    # ---------------------------------------------------------
    # VERIFY MISSING TENANT_ID
    # ---------------------------------------------------------

    try:
        bm25.refresh_index(
            documents=tenant_a_after_upload,
            tenant_id="",
        )

        raise AssertionError(
            "Missing tenant_id was not rejected."
        )

    except ValueError:
        print(
            "[PASS] Missing tenant_id rejected during "
            "BM25 refresh."
        )

    # ---------------------------------------------------------
    # VERIFY MIXED-TENANT DOCUMENTS
    # ---------------------------------------------------------

    mixed_documents = [
        tenant_a_initial[0],
        tenant_b_initial[0],
    ]

    try:
        bm25.refresh_index(
            documents=mixed_documents,
            tenant_id=tenant_a,
        )

        raise AssertionError(
            "Mixed-tenant documents were not rejected."
        )

    except ValueError:
        print(
            "[PASS] Mixed-tenant documents rejected "
            "during refresh."
        )

    # ---------------------------------------------------------
    # VERIFY SINGLETON BEHAVIOR
    # ---------------------------------------------------------

    bm25_second_reference = BM25Service()

    assert bm25_second_reference is bm25

    print(
        "[PASS] BM25Service uses the same process-local "
        "singleton instance."
    )

    assert bm25_second_reference.has_index(tenant_a)
    assert bm25_second_reference.has_index(tenant_b)

    print(
        "[PASS] BM25 indexes remain available through "
        "the second BM25Service reference."
    )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    bm25.clear()

    assert bm25.tenant_count() == 0

    print(
        "[PASS] BM25 indexes cleaned up."
    )

    print("\n" + "=" * 70)
    print("ALL BM25 REFRESH + TENANT ISOLATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
