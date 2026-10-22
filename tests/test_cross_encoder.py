from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.rag.reranking.cross_encoder import CrossEncoderReranker


def make_documents() -> list[Document]:
    return [
        Document(
            page_content="React and FastAPI experience.",
            metadata={
                "file_name": "resume.pdf",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Python and machine learning experience.",
            metadata={
                "file_name": "skills.pdf",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content="Advanced RAG and LLM projects.",
            metadata={
                "file_name": "projects.pdf",
                "chunk_index": 2,
            },
        ),
    ]


class FakeTokenizer:
    """Minimal tokenizer compatible with reranker diagnostics."""

    def __call__(
        self,
        query,
        document,
        truncation=True,
        max_length=None,
        return_length=True,
        **kwargs,
    ):
        token_count = len(
            query.split()
        ) + len(
            document.split()
        )

        return {
            "input_ids": list(range(token_count)),
        }


def build_reranker(
    scores,
    min_score_ratio: float = 0.0,
):
    """
    Build a CrossEncoderReranker without loading the
    real Hugging Face model.
    """

    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model_name = "test-reranker"
    reranker.min_score_ratio = min_score_ratio

    captured = {}

    def fake_predict(
        pairs,
        show_progress_bar=False,
    ):
        captured["pairs"] = pairs
        captured["show_progress_bar"] = show_progress_bar

        return scores

    reranker.model = SimpleNamespace(
        predict=fake_predict,
        tokenizer=FakeTokenizer(),
    )

    reranker._test_capture = captured

    return reranker


def test_rerank_empty_documents_returns_empty_list():
    reranker = build_reranker([])

    result = reranker.rerank(
        query="React experience",
        documents=[],
        top_k=5,
    )

    assert result == []


def test_rerank_top_k_zero_returns_empty_list():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.8, 0.7]
    )

    result = reranker.rerank(
        query="React experience",
        documents=documents,
        top_k=0,
    )

    assert result == []


def test_rerank_negative_top_k_returns_empty_list():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.8, 0.7]
    )

    result = reranker.rerank(
        query="React experience",
        documents=documents,
        top_k=-1,
    )

    assert result == []


def test_rerank_creates_correct_query_document_pairs():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.8, 0.7]
    )

    query = "What frontend technologies are used?"

    reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
    )

    assert reranker._test_capture["pairs"] == [
        (
            query,
            "React and FastAPI experience.",
        ),
        (
            query,
            "Python and machine learning experience.",
        ),
        (
            query,
            "Advanced RAG and LLM projects.",
        ),
    ]

    assert (
        reranker._test_capture["show_progress_bar"]
        is False
    )


def test_rerank_sorts_documents_by_descending_score():
    documents = make_documents()

    reranker = build_reranker(
        [0.2, 0.9, 0.5]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=3,
    )

    assert result[0] is documents[1]
    assert result[1] is documents[2]
    assert result[2] is documents[0]


def test_rerank_converts_scores_to_float():
    documents = make_documents()

    reranker = build_reranker(
        [1, 2.5, 0]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=3,
    )

    assert isinstance(
        result[0].metadata["rerank_score"],
        float,
    )

    assert isinstance(
        result[1].metadata["rerank_score"],
        float,
    )

    assert isinstance(
        result[2].metadata["rerank_score"],
        float,
    )


def test_rerank_stores_reranking_metadata():
    documents = make_documents()

    reranker = build_reranker(
        [0.8, 0.4, 0.2]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=3,
    )

    assert result[0].metadata["rerank_score"] == 0.8
    assert result[0].metadata["rerank_score_ratio"] == 1.0
    assert result[0].metadata["reranker_model"] == (
        "test-reranker"
    )

    assert result[1].metadata["rerank_score"] == 0.4
    assert result[1].metadata["rerank_score_ratio"] == 0.5
    assert result[1].metadata["reranker_model"] == (
        "test-reranker"
    )


def test_rerank_returns_only_top_k_documents():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.8, 0.7]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=2,
    )

    assert len(result) == 2
    assert result[0] is documents[0]
    assert result[1] is documents[1]


def test_rerank_rejects_mismatched_score_count():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.8]
    )

    with pytest.raises(
        RuntimeError,
        match="different number of scores",
    ):
        reranker.rerank(
            query="experience",
            documents=documents,
            top_k=3,
        )


def test_to_float_rejects_invalid_score():
    with pytest.raises(
        ValueError,
        match="Invalid cross-encoder score",
    ):
        CrossEncoderReranker._to_float(
            "not-a-number"
        )


def test_rerank_handles_zero_best_score():
    documents = make_documents()

    reranker = build_reranker(
        [0.0, 0.0, 0.0]
    )

    result = reranker.rerank(
        query="unrelated query",
        documents=documents,
        top_k=3,
    )

    assert len(result) == 3

    for document in result:
        assert document.metadata["rerank_score"] == 0.0
        assert document.metadata["rerank_score_ratio"] == 1.0
        assert document.metadata["reranker_model"] == (
            "test-reranker"
        )


def test_min_score_ratio_filters_low_scoring_documents():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.45, 0.2],
        min_score_ratio=0.5,
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=5,
    )

    assert len(result) == 2

    assert result[0] is documents[0]
    assert result[1] is documents[1]

    assert (
        documents[0].metadata["rerank_score_ratio"]
        == 1.0
    )

    assert (
        documents[1].metadata["rerank_score_ratio"]
        == 0.5
    )


def test_min_score_ratio_disabled_by_default():
    documents = make_documents()

    reranker = build_reranker(
        [0.9, 0.1, -0.5]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=5,
    )

    assert len(result) == 3


def test_rerank_preserves_document_objects_and_metadata():
    documents = make_documents()

    original_metadata = [
        document.metadata.copy()
        for document in documents
    ]

    reranker = build_reranker(
        [0.3, 0.9, 0.6]
    )

    result = reranker.rerank(
        query="experience",
        documents=documents,
        top_k=3,
    )

    returned_documents = set(
        id(document)
        for document in result
    )

    original_documents = set(
        id(document)
        for document in documents
    )

    assert returned_documents == original_documents

    for index, document in enumerate(documents):
        assert document.metadata["file_name"] == (
            original_metadata[index]["file_name"]
        )

        assert document.metadata["chunk_index"] == (
            original_metadata[index]["chunk_index"]
        )

        assert "rerank_score" in document.metadata
        assert "rerank_score_ratio" in document.metadata
        assert "reranker_model" in document.metadata