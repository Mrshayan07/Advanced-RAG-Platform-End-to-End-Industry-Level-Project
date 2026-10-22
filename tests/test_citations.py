from langchain_core.documents import Document

from app.rag.citations.models import (
    CitationClaim,
    CitationResponse,
)
from app.rag.citations.service import CitationService
from app.rag.citations.validator import CitationValidator


def test_citation_response_model():
    response = CitationResponse(
        claims=[
            CitationClaim(
                text="Shayan is a Full Stack Developer.",
                citation_ids=[1],
            )
        ]
    )

    assert len(response.claims) == 1
    assert response.claims[0].text == (
        "Shayan is a Full Stack Developer."
    )
    assert response.claims[0].citation_ids == [1]


def test_create_citations_from_documents():
    documents = [
        Document(
            page_content="First document content.",
            metadata={
                "file_name": "resume.pdf",
                "page_label": "1",
                "chunk_index": 0,
                "source": "storage/resume.pdf",
            },
        ),
        Document(
            page_content="Second document content.",
            metadata={
                "file_name": "experience.pdf",
                "page_label": "2",
                "chunk_index": 1,
                "source": "storage/experience.pdf",
            },
        ),
    ]

    service = CitationService()

    citations = service.create_citations(documents)

    assert len(citations) == 2

    assert citations[0].citation_id == 1
    assert citations[0].file_name == "resume.pdf"
    assert citations[0].page == "1"
    assert citations[0].chunk_index == 0
    assert citations[0].source == "storage/resume.pdf"

    assert citations[1].citation_id == 2
    assert citations[1].file_name == "experience.pdf"
    assert citations[1].page == "2"
    assert citations[1].chunk_index == 1
    assert citations[1].source == "storage/experience.pdf"


def test_create_citations_uses_defaults_for_missing_metadata():
    documents = [
        Document(
            page_content="Document without metadata.",
            metadata={},
        )
    ]

    service = CitationService()

    citations = service.create_citations(documents)

    assert len(citations) == 1
    assert citations[0].citation_id == 1
    assert citations[0].file_name == "Unknown"
    assert citations[0].page is None
    assert citations[0].chunk_index is None
    assert citations[0].source == ""


def test_format_context_contains_source_metadata_and_content():
    documents = [
        Document(
            page_content="Shayan works with React and FastAPI.",
            metadata={
                "file_name": "Shayan_CV.pdf",
                "page_label": "1",
                "chunk_index": 3,
            },
        ),
        Document(
            page_content="He also works with RAG systems.",
            metadata={
                "file_name": "Projects.pdf",
                "page_label": "2",
                "chunk_index": 5,
            },
        ),
    ]

    service = CitationService()

    context = service.format_context(documents)

    assert "[1]" in context
    assert "[2]" in context

    assert "Source: Shayan_CV.pdf" in context
    assert "Source: Projects.pdf" in context

    assert "Page: 1" in context
    assert "Page: 2" in context

    assert "Chunk: 3" in context
    assert "Chunk: 5" in context

    assert "Shayan works with React and FastAPI." in context
    assert "He also works with RAG systems." in context


def test_citation_validator_keeps_valid_ids_and_removes_invalid_ids():
    documents = [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
        Document(page_content="Document 3"),
    ]

    claims = [
        CitationClaim(
            text="Claim with valid citations.",
            citation_ids=[1, 3],
        ),
        CitationClaim(
            text="Claim with invalid citations.",
            citation_ids=[0, 2, 4, 99],
        ),
    ]

    validator = CitationValidator()

    validated_claims = validator.validate(
        claims=claims,
        documents=documents,
    )

    assert len(validated_claims) == 2

    assert validated_claims[0].citation_ids == [1, 3]
    assert validated_claims[1].citation_ids == [2]


def test_citation_validator_preserves_claim_text():
    documents = [
        Document(page_content="Document 1"),
    ]

    claim = CitationClaim(
        text="Original claim text must remain unchanged.",
        citation_ids=[1, 99],
    )

    validator = CitationValidator()

    validated_claims = validator.validate(
        claims=[claim],
        documents=documents,
    )

    assert validated_claims[0].text == (
        "Original claim text must remain unchanged."
    )
    assert validated_claims[0].citation_ids == [1]
