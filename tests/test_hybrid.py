from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.retrieval.hybrid import HybridRetriever

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def make_document(
    text: str,
    tenant_id: str = "tenant-a",
    file_name: str = "resume.pdf",
    chunk_index: int = 0,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "tenant_id": tenant_id,
            "file_name": file_name,
            "chunk_index": chunk_index,
        },
    )


def make_qdrant_point(
    text: str,
    tenant_id: str = "tenant-a",
    file_name: str = "resume.pdf",
    chunk_index: int = 0,
):
    return SimpleNamespace(
        id=f"{file_name}:{chunk_index}",
        payload={
            "text": text,
            "tenant_id": tenant_id,
            "file_name": file_name,
            "chunk_index": chunk_index,
        },
    )


def build_retriever(
    *,
    dense_documents=None,
    bm25_documents=None,
    fused_documents=None,
    reranked_documents=None,
):
    """
    Build HybridRetriever entirely with mocked dependencies.

    No real Qdrant, embeddings, BM25, query-rewriter,
    or cross-encoder is loaded.
    """

    bm25 = MagicMock()
    embeddings = MagicMock()
    qdrant = MagicMock()
    reranker = MagicMock()
    query_rewriter = MagicMock()

    retriever = HybridRetriever(
        bm25_service=bm25,
        embedding_service=embeddings,
        qdrant_service=qdrant,
        reranker=reranker,
        query_rewriter=query_rewriter,
        enable_logging=False,
    )

    if dense_documents is not None:
        embeddings.embed_query.return_value = [
            0.1,
            0.2,
            0.3,
        ]

        qdrant.search.return_value = [
            make_qdrant_point(
                document.page_content,
                tenant_id=document.metadata["tenant_id"],
                file_name=document.metadata.get(
                    "file_name",
                    "resume.pdf",
                ),
                chunk_index=document.metadata.get(
                    "chunk_index",
                    0,
                ),
            )
            for document in dense_documents
        ]

    if bm25_documents is not None:
        bm25.search.return_value = bm25_documents

    if fused_documents is not None:
        retriever.fusion = MagicMock()
        retriever.fusion.fuse.return_value = fused_documents

    if reranked_documents is not None:
        reranker.rerank.return_value = reranked_documents

    query_rewriter.rewrite.return_value = (
        "Automatically rewritten query"
    )

    return retriever


# ---------------------------------------------------------
# Constructor / Logging
# ---------------------------------------------------------


def test_constructor_stores_dependencies():
    bm25 = MagicMock()
    embeddings = MagicMock()
    qdrant = MagicMock()
    reranker = MagicMock()
    query_rewriter = MagicMock()

    retriever = HybridRetriever(
        bm25_service=bm25,
        embedding_service=embeddings,
        qdrant_service=qdrant,
        reranker=reranker,
        query_rewriter=query_rewriter,
        bm25_weight=0.3,
        dense_weight=0.7,
        enable_logging=False,
    )

    assert retriever.bm25 is bm25
    assert retriever.embeddings is embeddings
    assert retriever.qdrant is qdrant
    assert retriever.reranker is reranker
    assert retriever.query_rewriter is query_rewriter

    assert retriever.bm25_weight == 0.3
    assert retriever.dense_weight == 0.7

    assert retriever.enable_logging is False


def test_log_does_nothing_when_logging_disabled(capsys):
    retriever = build_retriever()

    retriever._log(
        "TEST",
        "message",
    )

    captured = capsys.readouterr()

    assert captured.out == ""


def test_log_prints_when_logging_enabled(capsys):
    retriever = build_retriever()
    retriever.enable_logging = True

    retriever._log(
        "TEST",
        "message",
    )

    captured = capsys.readouterr()

    assert "[HybridRetriever] TEST: message" in captured.out


# ---------------------------------------------------------
# Dense Retrieval
# ---------------------------------------------------------


def test_dense_retrieve_requires_tenant_id():
    retriever = build_retriever(
        dense_documents=[],
    )

    with pytest.raises(
        ValueError,
        match="tenant_id is required for dense retrieval",
    ):
        retriever._dense_retrieve(
            rewritten_query="query",
            dense_k=10,
            tenant_id="",
        )


def test_dense_retrieve_embeds_query():
    documents = [
        make_document(
            "React and FastAPI experience."
        )
    ]

    retriever = build_retriever(
        dense_documents=documents,
    )

    retriever._dense_retrieve(
        rewritten_query="frontend technologies",
        dense_k=5,
        tenant_id="tenant-a",
    )

    retriever.embeddings.embed_query.assert_called_once_with(
        "frontend technologies"
    )


def test_dense_retrieve_passes_query_vector_and_tenant_to_qdrant():
    documents = [
        make_document(
            "React and FastAPI experience."
        )
    ]

    retriever = build_retriever(
        dense_documents=documents,
    )

    retriever._dense_retrieve(
        rewritten_query="frontend technologies",
        dense_k=7,
        tenant_id="tenant-a",
    )

    retriever.qdrant.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        limit=7,
        tenant_id="tenant-a",
    )


def test_dense_retrieve_converts_qdrant_payload_to_documents():
    documents = [
        make_document(
            "React and FastAPI experience.",
            file_name="skills.pdf",
            chunk_index=2,
        )
    ]

    retriever = build_retriever(
        dense_documents=documents,
    )

    results = retriever._dense_retrieve(
        rewritten_query="frontend technologies",
        dense_k=5,
        tenant_id="tenant-a",
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        Document,
    )

    assert results[0].page_content == (
        "React and FastAPI experience."
    )

    assert results[0].metadata == {
        "tenant_id": "tenant-a",
        "file_name": "skills.pdf",
        "chunk_index": 2,
    }


def test_dense_retrieve_skips_qdrant_points_without_text():
    retriever = build_retriever(
        dense_documents=[],
    )

    retriever.qdrant.search.return_value = [
        make_qdrant_point(
            "",
            tenant_id="tenant-a",
        ),
        make_qdrant_point(
            "Valid document.",
            tenant_id="tenant-a",
        ),
    ]

    results = retriever._dense_retrieve(
        rewritten_query="query",
        dense_k=5,
        tenant_id="tenant-a",
    )

    assert len(results) == 1
    assert results[0].page_content == (
        "Valid document."
    )


def test_dense_retrieve_rejects_cross_tenant_document():
    retriever = build_retriever(
        dense_documents=[],
    )

    retriever.qdrant.search.return_value = [
        make_qdrant_point(
            "Tenant B document.",
            tenant_id="tenant-b",
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="Dense retrieval tenant isolation violation",
    ):
        retriever._dense_retrieve(
            rewritten_query="query",
            dense_k=5,
            tenant_id="tenant-a",
        )


def test_dense_retrieve_returns_empty_when_qdrant_returns_no_results():
    retriever = build_retriever(
        dense_documents=[],
    )

    retriever.qdrant.search.return_value = []

    results = retriever._dense_retrieve(
        rewritten_query="query",
        dense_k=5,
        tenant_id="tenant-a",
    )

    assert results == []


# ---------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------


def test_bm25_retrieve_requires_tenant_id():
    retriever = build_retriever(
        bm25_documents=[],
    )

    with pytest.raises(
        ValueError,
        match="tenant_id is required for BM25 retrieval",
    ):
        retriever._bm25_retrieve(
            rewritten_query="query",
            bm25_k=10,
            tenant_id="",
        )


def test_bm25_retrieve_calls_bm25_with_tenant_and_k():
    documents = [
        make_document(
            "Python and RAG experience."
        )
    ]

    retriever = build_retriever(
        bm25_documents=documents,
    )

    results = retriever._bm25_retrieve(
        rewritten_query="Python RAG",
        bm25_k=8,
        tenant_id="tenant-a",
    )

    retriever.bm25.search.assert_called_once_with(
        "Python RAG",
        tenant_id="tenant-a",
        k=8,
    )

    assert results == documents


def test_bm25_retrieve_rejects_cross_tenant_document():
    cross_tenant_document = make_document(
        "Tenant B document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        bm25_documents=[
            cross_tenant_document
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="BM25 retrieval tenant isolation violation",
    ):
        retriever._bm25_retrieve(
            rewritten_query="query",
            bm25_k=5,
            tenant_id="tenant-a",
        )


def test_bm25_retrieve_returns_empty_results():
    retriever = build_retriever(
        bm25_documents=[],
    )

    results = retriever._bm25_retrieve(
        rewritten_query="query",
        bm25_k=5,
        tenant_id="tenant-a",
    )

    assert results == []


# ---------------------------------------------------------
# Fusion
# ---------------------------------------------------------


def test_fuse_results_requires_tenant_id():
    retriever = build_retriever(
        fused_documents=[],
    )

    with pytest.raises(
        ValueError,
        match="tenant_id is required for RRF fusion",
    ):
        retriever._fuse_results(
            dense_documents=[],
            bm25_documents=[],
            fusion_top_k=5,
            tenant_id="",
        )


def test_fuse_results_calls_rrf_fusion():
    dense_documents = [
        make_document(
            "Dense document."
        )
    ]

    bm25_documents = [
        make_document(
            "BM25 document.",
            chunk_index=1,
        )
    ]

    fused_documents = [
        dense_documents[0],
        bm25_documents[0],
    ]

    retriever = build_retriever(
        fused_documents=fused_documents,
    )

    results = retriever._fuse_results(
        dense_documents=dense_documents,
        bm25_documents=bm25_documents,
        fusion_top_k=7,
        tenant_id="tenant-a",
    )

    retriever.fusion.fuse.assert_called_once_with(
        result_lists=[
            dense_documents,
            bm25_documents,
        ],
        top_k=7,
    )

    assert results == fused_documents


def test_fuse_results_rejects_cross_tenant_document():
    cross_tenant_document = make_document(
        "Tenant B document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        fused_documents=[
            cross_tenant_document
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="RRF fusion tenant isolation violation",
    ):
        retriever._fuse_results(
            dense_documents=[],
            bm25_documents=[],
            fusion_top_k=5,
            tenant_id="tenant-a",
        )


# ---------------------------------------------------------
# Tenant Validation
# ---------------------------------------------------------


def test_validate_tenant_documents_requires_tenant_id():
    retriever = build_retriever()

    with pytest.raises(
        ValueError,
        match="tenant_id is required for document validation",
    ):
        retriever._validate_tenant_documents(
            documents=[],
            tenant_id="",
            stage="TEST",
        )


def test_validate_tenant_documents_accepts_matching_documents():
    documents = [
        make_document(
            "Document one."
        ),
        make_document(
            "Document two.",
            chunk_index=1,
        ),
    ]

    retriever = build_retriever()

    result = retriever._validate_tenant_documents(
        documents=documents,
        tenant_id="tenant-a",
        stage="TEST",
    )

    assert result is None


def test_validate_tenant_documents_rejects_mismatched_document():
    documents = [
        make_document(
            "Tenant B document.",
            tenant_id="tenant-b",
        )
    ]

    retriever = build_retriever()

    with pytest.raises(
        RuntimeError,
        match="Tenant isolation violation at TEST",
    ):
        retriever._validate_tenant_documents(
            documents=documents,
            tenant_id="tenant-a",
            stage="TEST",
        )


# ---------------------------------------------------------
# search()
# ---------------------------------------------------------


def test_search_requires_tenant_id():
    retriever = build_retriever(
        dense_documents=[],
        bm25_documents=[],
        fused_documents=[],
        reranked_documents=[],
    )

    with pytest.raises(
        ValueError,
        match="tenant_id is required for hybrid retrieval",
    ):
        retriever.search(
            query="Question",
            tenant_id="",
        )


def test_search_automatically_rewrites_query():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.search(
        query="Original question",
        chat_history="Previous conversation",
        tenant_id="tenant-a",
    )

    retriever.query_rewriter.rewrite.assert_called_once_with(
        question="Original question",
        chat_history="Previous conversation",
    )


def test_search_uses_supplied_rewritten_query():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.search(
        query="Original question",
        rewritten_query="Supplied rewritten query",
        tenant_id="tenant-a",
    )

    retriever.query_rewriter.rewrite.assert_not_called()

    retriever.embeddings.embed_query.assert_called_once_with(
        "Supplied rewritten query"
    )


def test_search_passes_max_dense_bm25_k_to_fusion():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.search(
        query="Question",
        rewritten_query="Rewritten question",
        dense_k=7,
        bm25_k=12,
        top_k=3,
        tenant_id="tenant-a",
    )

    retriever.fusion.fuse.assert_called_once_with(
        result_lists=[
            [dense_document],
            [bm25_document],
        ],
        top_k=12,
    )


def test_search_reranks_fused_documents():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    fused_documents = [
        dense_document,
        bm25_document,
    ]

    reranked_documents = [
        bm25_document
    ]

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=fused_documents,
        reranked_documents=reranked_documents,
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten question",
        top_k=1,
        tenant_id="tenant-a",
    )

    retriever.reranker.rerank.assert_called_once_with(
        query="Rewritten question",
        documents=fused_documents,
        top_k=1,
    )

    assert result == reranked_documents


def test_search_normalizes_tenant_id_to_string():
    dense_document = make_document(
        "Dense document.",
        tenant_id="123",
    )

    bm25_document = make_document(
        "BM25 document.",
        tenant_id="123",
        chunk_index=1,
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document,
            bm25_document,
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id=123,
    )

    retriever.embeddings.embed_query.assert_called_once_with(
        "Rewritten"
    )

    retriever.qdrant.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        limit=10,
        tenant_id="123",
    )

    retriever.bm25.search.assert_called_once_with(
        "Rewritten",
        tenant_id="123",
        k=10,
    )


def test_search_returns_empty_when_both_retrievers_return_empty():
    retriever = build_retriever(
        dense_documents=[],
        bm25_documents=[],
        fused_documents=[],
        reranked_documents=[],
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id="tenant-a",
    )

    assert result == []


def test_search_falls_back_when_dense_retrieval_fails():
    bm25_document = make_document(
        "BM25 document."
    )

    retriever = build_retriever(
        dense_documents=[],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            bm25_document
        ],
        reranked_documents=[
            bm25_document
        ],
    )

    retriever.embeddings.embed_query.side_effect = (
        RuntimeError("Embedding service unavailable")
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id="tenant-a",
    )

    assert result == [
        bm25_document
    ]


def test_search_falls_back_when_bm25_retrieval_fails():
    dense_document = make_document(
        "Dense document."
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.bm25.search.side_effect = (
        RuntimeError("BM25 unavailable")
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id="tenant-a",
    )

    assert result == [
        dense_document
    ]


def test_search_falls_back_to_fused_documents_when_reranker_fails():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    fused_documents = [
        dense_document,
        bm25_document,
    ]

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=fused_documents,
    )

    retriever.reranker.rerank.side_effect = (
        RuntimeError("Reranker unavailable")
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        top_k=2,
        tenant_id="tenant-a",
    )

    assert result == fused_documents


def test_search_reranker_failure_respects_top_k():
    documents = [
        make_document(
            "Document one.",
            chunk_index=0,
        ),
        make_document(
            "Document two.",
            chunk_index=1,
        ),
        make_document(
            "Document three.",
            chunk_index=2,
        ),
    ]

    retriever = build_retriever(
        dense_documents=documents,
        bm25_documents=documents,
        fused_documents=documents,
    )

    retriever.reranker.rerank.side_effect = (
        RuntimeError("Reranker unavailable")
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        top_k=2,
        tenant_id="tenant-a",
    )

    assert result == documents[:2]


def test_search_reranker_tenant_violation_uses_fallback():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    fused_documents = [
        dense_document,
        bm25_document,
    ]

    invalid_reranked_document = make_document(
        "Invalid tenant document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=fused_documents,
        reranked_documents=[
            invalid_reranked_document
        ],
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        top_k=1,
        tenant_id="tenant-a",
    )

    assert result == fused_documents[:1]


def test_search_dense_tenant_validation_failure_is_caught():
    dense_document = make_document(
        "Invalid dense document.",
        tenant_id="tenant-b",
    )

    bm25_document = make_document(
        "Valid BM25 document.",
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            bm25_document
        ],
        reranked_documents=[
            bm25_document
        ],
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id="tenant-a",
    )

    assert result == [
        bm25_document
    ]


def test_search_bm25_tenant_validation_failure_is_caught():
    dense_document = make_document(
        "Valid dense document."
    )

    bm25_document = make_document(
        "Invalid BM25 document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    result = retriever.search(
        query="Question",
        rewritten_query="Rewritten",
        tenant_id="tenant-a",
    )

    assert result == [
        dense_document
    ]


# ---------------------------------------------------------
# search_with_scores()
# ---------------------------------------------------------


def test_search_with_scores_requires_tenant_id():
    retriever = build_retriever()

    with pytest.raises(
        ValueError,
        match="tenant_id is required for hybrid retrieval",
    ):
        retriever.search_with_scores(
            query="Question",
            tenant_id="",
        )


def test_search_with_scores_rewrites_query():
    dense_document = make_document(
        "Dense document."
    )

    bm25_document = make_document(
        "BM25 document.",
        chunk_index=1,
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            dense_document
        ],
    )

    retriever.search_with_scores(
        query="Original question",
        chat_history="Previous conversation",
        tenant_id="tenant-a",
    )

    retriever.query_rewriter.rewrite.assert_called_once_with(
        question="Original question",
        chat_history="Previous conversation",
    )


def test_search_with_scores_returns_float_scores():
    document = make_document(
        "Relevant document."
    )

    document.metadata["rerank_score"] = 0.87

    retriever = build_retriever(
        dense_documents=[
            document
        ],
        bm25_documents=[
            document
        ],
        fused_documents=[
            document
        ],
        reranked_documents=[
            document
        ],
    )

    result = retriever.search_with_scores(
        query="Question",
        top_k=1,
        tenant_id="tenant-a",
    )

    assert len(result) == 1

    returned_document, score = result[0]

    assert returned_document is document
    assert isinstance(score, float)
    assert score == 0.87


def test_search_with_scores_defaults_missing_score_to_zero():
    document = make_document(
        "Relevant document."
    )

    retriever = build_retriever(
        dense_documents=[
            document
        ],
        bm25_documents=[
            document
        ],
        fused_documents=[
            document
        ],
        reranked_documents=[
            document
        ],
    )

    result = retriever.search_with_scores(
        query="Question",
        top_k=1,
        tenant_id="tenant-a",
    )

    assert result[0][1] == 0.0
    assert isinstance(
        result[0][1],
        float,
    )


def test_search_with_scores_passes_top_k_to_reranker():
    documents = [
        make_document(
            "Document one.",
            chunk_index=0,
        ),
        make_document(
            "Document two.",
            chunk_index=1,
        ),
    ]

    retriever = build_retriever(
        dense_documents=documents,
        bm25_documents=documents,
        fused_documents=documents,
        reranked_documents=documents[:1],
    )

    retriever.search_with_scores(
        query="Question",
        top_k=1,
        tenant_id="tenant-a",
    )

    retriever.reranker.rerank.assert_called_once_with(
        query="Automatically rewritten query",
        documents=documents,
        top_k=1,
    )


def test_search_with_scores_rejects_cross_tenant_dense_documents():
    cross_tenant_document = make_document(
        "Tenant B document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        dense_documents=[
            cross_tenant_document
        ],
        bm25_documents=[],
        fused_documents=[],
        reranked_documents=[],
    )

    with pytest.raises(
        RuntimeError,
        match="Dense retrieval tenant isolation violation",
    ):
        retriever.search_with_scores(
            query="Question",
            tenant_id="tenant-a",
        )


def test_search_with_scores_rejects_cross_tenant_reranked_documents():
    dense_document = make_document(
        "Valid dense document."
    )

    bm25_document = make_document(
        "Valid BM25 document.",
        chunk_index=1,
    )

    invalid_reranked_document = make_document(
        "Tenant B document.",
        tenant_id="tenant-b",
    )

    retriever = build_retriever(
        dense_documents=[
            dense_document
        ],
        bm25_documents=[
            bm25_document
        ],
        fused_documents=[
            dense_document
        ],
        reranked_documents=[
            invalid_reranked_document
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Tenant isolation violation at RERANKING",
    ):
        retriever.search_with_scores(
            query="Question",
            tenant_id="tenant-a",
        )