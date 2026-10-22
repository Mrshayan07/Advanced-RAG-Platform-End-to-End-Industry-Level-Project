from langchain_core.documents import Document

from app.rag.retrieval.fusion import RRFFusion


def make_document(
    file_name: str,
    chunk_index: int,
    content: str,
    source: str = "",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "file_name": file_name,
            "chunk_index": chunk_index,
            "source": source,
        },
    )


def test_fuse_empty_result_lists_returns_empty_list():
    fusion = RRFFusion()

    result = fusion.fuse(
        result_lists=[],
        top_k=5,
    )

    assert result == []


def test_fuse_single_result_list_preserves_ranking():
    documents = [
        make_document(
            "doc1.pdf",
            0,
            "First document",
        ),
        make_document(
            "doc2.pdf",
            0,
            "Second document",
        ),
        make_document(
            "doc3.pdf",
            0,
            "Third document",
        ),
    ]

    fusion = RRFFusion()

    result = fusion.fuse(
        result_lists=[documents],
        top_k=3,
    )

    assert result == documents


def test_fuse_calculates_rrf_scores_correctly():
    doc_a = make_document(
        "a.pdf",
        0,
        "Document A",
    )

    doc_b = make_document(
        "b.pdf",
        0,
        "Document B",
    )

    fusion = RRFFusion(k=60)

    result = fusion.fuse(
        result_lists=[
            [doc_a, doc_b],
        ],
        top_k=2,
    )

    expected_order = [
        doc_a,
        doc_b,
    ]

    assert result == expected_order

    score_a = 1 / (60 + 1)
    score_b = 1 / (60 + 2)

    assert score_a > score_b


def test_document_appearing_in_multiple_lists_gets_boosted():
    doc_a = make_document(
        "a.pdf",
        0,
        "Document A",
    )

    doc_b = make_document(
        "b.pdf",
        0,
        "Document B",
    )

    doc_c = make_document(
        "c.pdf",
        0,
        "Document C",
    )

    fusion = RRFFusion(k=60)

    result = fusion.fuse(
        result_lists=[
            [doc_a, doc_b],
            [doc_b, doc_c],
        ],
        top_k=3,
    )

    assert result[0].page_content == "Document B"


def test_duplicate_document_appears_only_once():
    doc_a = make_document(
        "a.pdf",
        0,
        "Document A",
    )

    result = RRFFusion().fuse(
        result_lists=[
            [doc_a],
            [doc_a],
        ],
        top_k=5,
    )

    assert len(result) == 1
    assert result[0].page_content == "Document A"


def test_top_k_limits_number_of_results():
    documents = [
        make_document(
            f"doc{i}.pdf",
            0,
            f"Document {i}",
        )
        for i in range(5)
    ]

    result = RRFFusion().fuse(
        result_lists=[documents],
        top_k=2,
    )

    assert len(result) == 2

    assert result[0].page_content == "Document 0"
    assert result[1].page_content == "Document 1"


def test_first_document_version_is_preserved():
    first_version = make_document(
        "resume.pdf",
        0,
        "First version of document",
        source="path/a/resume.pdf",
    )

    second_version = make_document(
        "resume.pdf",
        0,
        "Second version of document",
        source="path/b/resume.pdf",
    )

    result = RRFFusion().fuse(
        result_lists=[
            [first_version],
            [second_version],
        ],
        top_k=5,
    )

    assert len(result) == 1

    assert result[0].page_content == (
        "First version of document"
    )

    assert result[0].metadata["source"] == (
        "path/a/resume.pdf"
    )


def test_same_file_and_chunk_are_same_logical_document():
    document_from_bm25 = make_document(
        "resume.pdf",
        3,
        "Resume chunk",
        source="relative/resume.pdf",
    )

    document_from_qdrant = make_document(
        "resume.pdf",
        3,
        "Resume chunk",
        source="C:/absolute/resume.pdf",
    )

    result = RRFFusion().fuse(
        result_lists=[
            [document_from_bm25],
            [document_from_qdrant],
        ],
        top_k=5,
    )

    assert len(result) == 1


def test_different_chunks_are_different_documents():
    chunk_zero = make_document(
        "resume.pdf",
        0,
        "Chunk zero",
    )

    chunk_one = make_document(
        "resume.pdf",
        1,
        "Chunk one",
    )

    result = RRFFusion().fuse(
        result_lists=[
            [chunk_zero],
            [chunk_one],
        ],
        top_k=5,
    )

    assert len(result) == 2

    assert {
        document.metadata["chunk_index"]
        for document in result
    } == {0, 1}


def test_custom_rrf_k_changes_score_values():
    doc_a = make_document(
        "a.pdf",
        0,
        "Document A",
    )

    doc_b = make_document(
        "b.pdf",
        0,
        "Document B",
    )

    fusion_default = RRFFusion(k=60)
    fusion_custom = RRFFusion(k=10)

    result_default = fusion_default.fuse(
        result_lists=[
            [doc_a, doc_b],
        ],
        top_k=2,
    )

    result_custom = fusion_custom.fuse(
        result_lists=[
            [doc_a, doc_b],
        ],
        top_k=2,
    )

    assert result_default[0] == doc_a
    assert result_default[1] == doc_b

    assert result_custom[0] == doc_a
    assert result_custom[1] == doc_b

    default_score_a = 1 / (60 + 1)
    custom_score_a = 1 / (10 + 1)

    assert custom_score_a > default_score_a