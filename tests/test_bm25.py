import pytest
from langchain_core.documents import Document

from app.rag.retrieval.bm25 import BM25Service


@pytest.fixture(autouse=True)
def clean_bm25_service():
    """
    Ensure every BM25 test starts with a clean
    process-local BM25 service.
    """

    service = BM25Service()
    service.clear()

    yield

    service.clear()


def make_documents(
    tenant_id: str,
) -> list[Document]:
    return [
        Document(
            page_content=(
                "Shayan is a Full Stack Developer "
                "working with React and FastAPI."
            ),
            metadata={
                "tenant_id": tenant_id,
                "file_name": "resume.pdf",
            },
        ),
        Document(
            page_content=(
                "Shayan has experience with "
                "Python, SQL, machine learning, and RAG."
            ),
            metadata={
                "tenant_id": tenant_id,
                "file_name": "skills.pdf",
            },
        ),
        Document(
            page_content=(
                "Shayan builds advanced retrieval "
                "augmented generation systems."
            ),
            metadata={
                "tenant_id": tenant_id,
                "file_name": "projects.pdf",
            },
        ),
    ]


def test_bm25_service_is_singleton():
    service_one = BM25Service()
    service_two = BM25Service()

    assert service_one is service_two


def test_build_index_creates_tenant_index():
    service = BM25Service()

    documents = make_documents("tenant-a")

    service.build_index(
        documents=documents,
        tenant_id="tenant-a",
    )

    assert service.has_index("tenant-a") is True
    assert service.tenant_count() == 1


def test_build_index_rejects_missing_tenant_id():
    service = BM25Service()

    documents = make_documents("tenant-a")

    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        service.build_index(
            documents=documents,
            tenant_id="",
        )


def test_build_index_rejects_empty_documents():
    service = BM25Service()

    with pytest.raises(
        ValueError,
        match="Cannot build BM25 index with no documents",
    ):
        service.build_index(
            documents=[],
            tenant_id="tenant-a",
        )


def test_build_index_rejects_document_without_tenant_id():
    service = BM25Service()

    documents = [
        Document(
            page_content="Document without tenant metadata.",
            metadata={},
        )
    ]

    with pytest.raises(
        ValueError,
        match="Every BM25 document must contain tenant_id",
    ):
        service.build_index(
            documents=documents,
            tenant_id="tenant-a",
        )


def test_build_index_rejects_tenant_mismatch():
    service = BM25Service()

    documents = [
        Document(
            page_content="Tenant A document.",
            metadata={
                "tenant_id": "tenant-b",
            },
        )
    ]

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        service.build_index(
            documents=documents,
            tenant_id="tenant-a",
        )


def test_search_returns_relevant_documents():
    service = BM25Service()

    documents = make_documents("tenant-a")

    service.build_index(
        documents=documents,
        tenant_id="tenant-a",
    )

    results = service.search(
        query="React FastAPI",
        tenant_id="tenant-a",
        k=2,
    )

    assert len(results) == 2

    assert all(
        document.metadata["tenant_id"] == "tenant-a"
        for document in results
    )

    assert any(
        "React" in document.page_content
        or "FastAPI" in document.page_content
        for document in results
    )


def test_search_rejects_missing_tenant_id():
    service = BM25Service()

    documents = make_documents("tenant-a")

    service.build_index(
        documents=documents,
        tenant_id="tenant-a",
    )

    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        service.search(
            query="React",
            tenant_id="",
        )


def test_search_requires_initialized_tenant_index():
    service = BM25Service()

    with pytest.raises(
        RuntimeError,
        match="BM25 index is not initialized",
    ):
        service.search(
            query="React",
            tenant_id="tenant-a",
        )


def test_search_preserves_tenant_isolation():
    service = BM25Service()

    tenant_a_documents = [
        Document(
            page_content="Tenant A uses React and FastAPI.",
            metadata={
                "tenant_id": "tenant-a",
            },
        )
    ]

    tenant_b_documents = [
        Document(
            page_content="Tenant B uses Angular and Django.",
            metadata={
                "tenant_id": "tenant-b",
            },
        )
    ]

    service.build_index(
        documents=tenant_a_documents,
        tenant_id="tenant-a",
    )

    service.build_index(
        documents=tenant_b_documents,
        tenant_id="tenant-b",
    )

    tenant_a_results = service.search(
        query="Angular Django",
        tenant_id="tenant-a",
        k=5,
    )

    assert all(
        document.metadata["tenant_id"] == "tenant-a"
        for document in tenant_a_results
    )

    assert not any(
        "Angular" in document.page_content
        or "Django" in document.page_content
        for document in tenant_a_results
    )


def test_refresh_index_replaces_existing_tenant_index():
    service = BM25Service()

    original_documents = [
        Document(
            page_content="Original tenant document about React.",
            metadata={
                "tenant_id": "tenant-a",
            },
        )
    ]

    refreshed_documents = [
        Document(
            page_content="Refreshed tenant document about Python.",
            metadata={
                "tenant_id": "tenant-a",
            },
        )
    ]

    service.build_index(
        documents=original_documents,
        tenant_id="tenant-a",
    )

    service.refresh_index(
        documents=refreshed_documents,
        tenant_id="tenant-a",
    )

    results = service.search(
        query="Python",
        tenant_id="tenant-a",
        k=5,
    )

    assert len(results) == 1
    assert results[0].page_content == (
        "Refreshed tenant document about Python."
    )


def test_remove_tenant_removes_only_requested_index():
    service = BM25Service()

    service.build_index(
        documents=make_documents("tenant-a"),
        tenant_id="tenant-a",
    )

    service.build_index(
        documents=make_documents("tenant-b"),
        tenant_id="tenant-b",
    )

    assert service.tenant_count() == 2

    service.remove_tenant("tenant-a")

    assert service.has_index("tenant-a") is False
    assert service.has_index("tenant-b") is True
    assert service.tenant_count() == 1


def test_clear_removes_all_indexes():
    service = BM25Service()

    service.build_index(
        documents=make_documents("tenant-a"),
        tenant_id="tenant-a",
    )

    service.build_index(
        documents=make_documents("tenant-b"),
        tenant_id="tenant-b",
    )

    assert service.tenant_count() == 2

    service.clear()

    assert service.tenant_count() == 0
    assert service.has_index("tenant-a") is False
    assert service.has_index("tenant-b") is False