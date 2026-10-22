from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.citations.models import (
    Citation,
    CitationClaim,
    CitationResponse,
)
from app.rag.graph.nodes import RAGNodes

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


def make_claim(
    text: str = "Shayan is a Full Stack Developer.",
    citation_ids: list[int] | None = None,
) -> CitationClaim:
    return CitationClaim(
        text=text,
        citation_ids=(
            citation_ids
            if citation_ids is not None
            else [1]
        ),
    )


def make_citation_response(
    claims: list[CitationClaim] | None = None,
) -> CitationResponse:
    return CitationResponse(
        claims=(
            claims
            if claims is not None
            else [make_claim()]
        )
    )


def build_nodes() -> RAGNodes:
    """
    Build RAGNodes without initializing real Gemini/Groq clients.

    All external dependencies are replaced with mocks.
    """

    nodes = RAGNodes.__new__(RAGNodes)

    nodes.query_rewriter = MagicMock()
    nodes.retriever = MagicMock()

    nodes.citation_service = MagicMock()
    nodes.citation_validator = MagicMock()

    nodes.gemini_llm = MagicMock()
    nodes.groq_llm = MagicMock()

    nodes.gemini_chain = MagicMock()
    nodes.groq_chain = MagicMock()

    return nodes


# ---------------------------------------------------------
# rewrite()
# ---------------------------------------------------------


def test_rewrite_calls_query_rewriter_with_question_and_history():
    nodes = build_nodes()

    nodes.query_rewriter.rewrite.return_value = (
        "Shayan frontend development technologies"
    )

    state = {
        "question": (
            "What technologies did Shayan use "
            "for frontend development?"
        ),
        "chat_history": (
            "Previous question about Shayan."
        ),
    }

    result = nodes.rewrite(state)

    nodes.query_rewriter.rewrite.assert_called_once_with(
        question=state["question"],
        chat_history=state["chat_history"],
    )

    assert result["rewritten_query"] == (
        "Shayan frontend development technologies"
    )


def test_rewrite_uses_empty_history_when_missing():
    nodes = build_nodes()

    nodes.query_rewriter.rewrite.return_value = (
        "Shayan role"
    )

    state = {
        "question": "What is Shayan's role?"
    }

    result = nodes.rewrite(state)

    nodes.query_rewriter.rewrite.assert_called_once_with(
        question="What is Shayan's role?",
        chat_history="",
    )

    assert result["rewritten_query"] == (
        "Shayan role"
    )


def test_rewrite_preserves_existing_state():
    nodes = build_nodes()

    nodes.query_rewriter.rewrite.return_value = (
        "Rewritten question"
    )

    state = {
        "tenant_id": "tenant-a",
        "question": "Original question",
        "chat_history": "History",
        "user_memory": [
            {
                "key": "name",
                "value": "Shayan",
                "memory_type": "profile",
            }
        ],
    }

    result = nodes.rewrite(state)

    assert result["tenant_id"] == "tenant-a"
    assert result["question"] == "Original question"
    assert result["chat_history"] == "History"
    assert result["user_memory"] == state["user_memory"]
    assert result["rewritten_query"] == (
        "Rewritten question"
    )


# ---------------------------------------------------------
# retrieve()
# ---------------------------------------------------------


def test_retrieve_requires_tenant_id():
    nodes = build_nodes()

    state = {
        "question": "What technologies does Shayan know?",
    }

    with pytest.raises(
        ValueError,
        match="tenant_id is required for RAG retrieval",
    ):
        nodes.retrieve(state)


def test_retrieve_requires_non_empty_tenant_id():
    nodes = build_nodes()

    state = {
        "question": "What technologies does Shayan know?",
        "tenant_id": "",
    }

    with pytest.raises(
        ValueError,
        match="tenant_id is required for RAG retrieval",
    ):
        nodes.retrieve(state)


def test_retrieve_calls_hybrid_retriever_with_expected_parameters():
    nodes = build_nodes()

    documents = [
        make_document(
            "Shayan works with React."
        ),
        make_document(
            "Shayan works with FastAPI.",
            chunk_index=1,
        ),
    ]

    nodes.retriever.search.return_value = documents

    state = {
        "tenant_id": "tenant-a",
        "question": "What technologies does Shayan use?",
        "rewritten_query": "Shayan technologies",
        "chat_history": "Previous conversation",
    }

    result = nodes.retrieve(state)

    nodes.retriever.search.assert_called_once_with(
        query="What technologies does Shayan use?",
        chat_history="Previous conversation",
        rewritten_query="Shayan technologies",
        dense_k=10,
        bm25_k=10,
        top_k=3,
        tenant_id="tenant-a",
    )

    assert result["documents"] == documents


def test_retrieve_uses_empty_chat_history_when_missing():
    nodes = build_nodes()

    nodes.retriever.search.return_value = []

    state = {
        "tenant_id": "tenant-a",
        "question": "What is Shayan's role?",
    }

    result = nodes.retrieve(state)

    nodes.retriever.search.assert_called_once_with(
        query="What is Shayan's role?",
        chat_history="",
        rewritten_query=None,
        dense_k=10,
        bm25_k=10,
        top_k=3,
        tenant_id="tenant-a",
    )

    assert result["documents"] == []


def test_retrieve_preserves_existing_state():
    nodes = build_nodes()

    documents = [
        make_document("Relevant document.")
    ]

    nodes.retriever.search.return_value = documents

    state = {
        "tenant_id": "tenant-a",
        "question": "Question",
        "rewritten_query": "Rewritten",
        "chat_history": "History",
        "user_memory": [
            {
                "key": "name",
                "value": "Shayan",
                "memory_type": "profile",
            }
        ],
    }

    result = nodes.retrieve(state)

    assert result["tenant_id"] == "tenant-a"
    assert result["question"] == "Question"
    assert result["rewritten_query"] == "Rewritten"
    assert result["chat_history"] == "History"
    assert result["user_memory"] == state["user_memory"]
    assert result["documents"] == documents


# ---------------------------------------------------------
# _extract_response_text()
# ---------------------------------------------------------


def test_extract_response_text_from_string():
    nodes = build_nodes()

    result = nodes._extract_response_text(
        "  Hello from the LLM.  "
    )

    assert result == "Hello from the LLM."


def test_extract_response_text_from_text_blocks():
    nodes = build_nodes()

    content = [
        {
            "type": "text",
            "text": "First part.",
        },
        {
            "type": "text",
            "text": "Second part.",
        },
    ]

    result = nodes._extract_response_text(content)

    assert result == (
        "First part.\nSecond part."
    )


def test_extract_response_text_from_dict_with_text_key():
    nodes = build_nodes()

    content = [
        {
            "text": "First part.",
        },
        {
            "text": "Second part.",
        },
    ]

    result = nodes._extract_response_text(content)

    assert result == (
        "First part.\nSecond part."
    )


def test_extract_response_text_from_objects_with_text_attribute():
    nodes = build_nodes()

    content = [
        SimpleNamespace(
            text="First part."
        ),
        SimpleNamespace(
            text="Second part."
        ),
    ]

    result = nodes._extract_response_text(content)

    assert result == (
        "First part.\nSecond part."
    )


def test_extract_response_text_ignores_non_text_blocks():
    nodes = build_nodes()

    content = [
        {
            "type": "image",
            "url": "image.png",
        },
        {
            "type": "text",
            "text": "Valid text.",
        },
        SimpleNamespace(
            value="Not text"
        ),
    ]

    result = nodes._extract_response_text(content)

    assert result == "Valid text."


def test_extract_response_text_from_empty_list():
    nodes = build_nodes()

    result = nodes._extract_response_text([])

    assert result == ""


def test_extract_response_text_converts_other_types_to_string():
    nodes = build_nodes()

    result = nodes._extract_response_text(
        12345
    )

    assert result == "12345"


# ---------------------------------------------------------
# _is_retryable_llm_error()
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "error_message",
    [
        "HTTP 429 Too Many Requests",
        "HTTP 500 Internal Server Error",
        "HTTP 502 Bad Gateway",
        "HTTP 503 Service Unavailable",
        "HTTP 504 Gateway Timeout",
        "Resource exhausted",
        "Rate limit exceeded",
        "rate_limit reached",
        "Too many requests",
        "Connection timeout",
        "Request timed out",
        "Deadline exceeded",
        "Temporarily unavailable",
        "Temporary service failure",
        "Service unavailable",
    ],
)
def test_retryable_llm_errors_are_detected(
    error_message,
):
    nodes = build_nodes()

    assert nodes._is_retryable_llm_error(
        RuntimeError(error_message)
    ) is True


@pytest.mark.parametrize(
    "error_message",
    [
        "Invalid API key",
        "Authentication failed",
        "Malformed request",
        "Permission denied",
        "Invalid model name",
        "Bad structured output",
    ],
)
def test_non_retryable_llm_errors_are_detected(
    error_message,
):
    nodes = build_nodes()

    assert nodes._is_retryable_llm_error(
        RuntimeError(error_message)
    ) is False


# ---------------------------------------------------------
# _invoke_llm_with_fallback()
# ---------------------------------------------------------


def test_invoke_llm_returns_gemini_response_on_first_attempt():
    nodes = build_nodes()

    expected_response = make_citation_response()

    nodes.gemini_chain.invoke.return_value = (
        expected_response
    )

    payload = {
        "question": "What is Shayan's role?",
        "context": "Shayan is a Full Stack Developer.",
    }

    result = nodes._invoke_llm_with_fallback(
        payload
    )

    assert result is expected_response

    nodes.gemini_chain.invoke.assert_called_once_with(
        payload
    )

    nodes.groq_chain.invoke.assert_not_called()


def test_invoke_llm_retries_retryable_gemini_error():
    nodes = build_nodes()

    expected_response = make_citation_response()

    nodes.gemini_chain.invoke.side_effect = [
        RuntimeError("HTTP 503 Service Unavailable"),
        RuntimeError("HTTP 503 Service Unavailable"),
        expected_response,
    ]

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ) as sleep_mock:
        result = nodes._invoke_llm_with_fallback(
            payload
        )

    assert result is expected_response

    assert (
        nodes.gemini_chain.invoke.call_count
        == 3
    )

    sleep_mock.assert_any_call(1)
    sleep_mock.assert_any_call(2)

    assert sleep_mock.call_count == 2

    nodes.groq_chain.invoke.assert_not_called()


def test_invoke_llm_uses_exponential_backoff():
    nodes = build_nodes()

    expected_response = make_citation_response()

    nodes.gemini_chain.invoke.side_effect = [
        RuntimeError("HTTP 503"),
        RuntimeError("HTTP 503"),
        RuntimeError("HTTP 503"),
    ]

    nodes.groq_chain.invoke.return_value = (
        expected_response
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ) as sleep_mock:
        result = nodes._invoke_llm_with_fallback(
            payload
        )

    assert result is expected_response

    assert sleep_mock.call_args_list == [
        ((1,),),
        ((2,),),
    ]


def test_invoke_llm_falls_back_to_groq_after_gemini_retries():
    nodes = build_nodes()

    expected_response = make_citation_response()

    nodes.gemini_chain.invoke.side_effect = (
        RuntimeError("HTTP 503")
    )

    nodes.groq_chain.invoke.return_value = (
        expected_response
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ) as sleep_mock:
        result = nodes._invoke_llm_with_fallback(
            payload
        )

    assert result is expected_response

    assert (
        nodes.gemini_chain.invoke.call_count
        == 3
    )

    sleep_mock.assert_has_calls(
        [
            ((1,),),
            ((2,),),
        ]
    )

    nodes.groq_chain.invoke.assert_called_once_with(
        payload
    )


def test_invoke_llm_does_not_retry_non_retryable_gemini_error():
    nodes = build_nodes()

    nodes.gemini_chain.invoke.side_effect = (
        RuntimeError("Invalid API key")
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ) as sleep_mock:
        with pytest.raises(
            RuntimeError,
            match="Invalid API key",
        ):
            nodes._invoke_llm_with_fallback(
                payload
            )

    nodes.gemini_chain.invoke.assert_called_once_with(
        payload
    )

    nodes.groq_chain.invoke.assert_not_called()

    sleep_mock.assert_not_called()


def test_invoke_llm_rejects_invalid_gemini_structured_output():
    nodes = build_nodes()

    nodes.gemini_chain.invoke.return_value = (
        {"claims": []}
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with pytest.raises(
        ValueError,
        match="unexpected structured output type",
    ):
        nodes._invoke_llm_with_fallback(
            payload
        )

    nodes.gemini_chain.invoke.assert_called_once_with(
        payload
    )

    nodes.groq_chain.invoke.assert_not_called()


def test_invoke_llm_rejects_invalid_groq_structured_output():
    nodes = build_nodes()

    nodes.gemini_chain.invoke.side_effect = (
        RuntimeError("HTTP 503")
    )

    nodes.groq_chain.invoke.return_value = (
        {"claims": []}
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ):
        with pytest.raises(
            RuntimeError,
            match="Both Gemini and Groq structured generation failed",
        ):
            nodes._invoke_llm_with_fallback(
                payload
            )

    assert (
        nodes.gemini_chain.invoke.call_count
        == 3
    )

    nodes.groq_chain.invoke.assert_called_once_with(
        payload
    )


def test_invoke_llm_raises_runtime_error_when_groq_fails():
    nodes = build_nodes()

    nodes.gemini_chain.invoke.side_effect = (
        RuntimeError("HTTP 503")
    )

    nodes.groq_chain.invoke.side_effect = (
        RuntimeError("Groq unavailable")
    )

    payload = {
        "question": "Question",
        "context": "Context",
    }

    with patch(
        "app.rag.graph.nodes.time.sleep"
    ):
        with pytest.raises(
            RuntimeError,
            match="Both Gemini and Groq structured generation failed",
        ) as exc_info:
            nodes._invoke_llm_with_fallback(
                payload
            )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    assert str(
        exc_info.value.__cause__
    ) == "Groq unavailable"


# ---------------------------------------------------------
# _parse_citation_response()
# ---------------------------------------------------------


def test_parse_citation_response_parses_pure_json():
    nodes = build_nodes()

    content = """
    {
        "claims": [
            {
                "text": "Shayan is a Full Stack Developer.",
                "citation_ids": [1]
            }
        ]
    }
    """

    result = nodes._parse_citation_response(
        content
    )

    assert isinstance(
        result,
        CitationResponse,
    )

    assert len(result.claims) == 1

    assert result.claims[0].text == (
        "Shayan is a Full Stack Developer."
    )

    assert result.claims[0].citation_ids == [1]


def test_parse_citation_response_parses_json_code_fence():
    nodes = build_nodes()

    content = """```json
{
    "claims": [
        {
            "text": "Shayan uses React.",
            "citation_ids": [1]
        }
    ]
}
```"""

    result = nodes._parse_citation_response(
        content
    )

    assert isinstance(
        result,
        CitationResponse,
    )

    assert result.claims[0].text == (
        "Shayan uses React."
    )

    assert result.claims[0].citation_ids == [1]


def test_parse_citation_response_parses_json_embedded_in_text():
    nodes = build_nodes()

    content = """
    Here is the structured response:

    {
        "claims": [
            {
                "text": "Shayan works with FastAPI.",
                "citation_ids": [2]
            }
        ]
    }

    End of response.
    """

    result = nodes._parse_citation_response(
        content
    )

    assert isinstance(
        result,
        CitationResponse,
    )

    assert result.claims[0].text == (
        "Shayan works with FastAPI."
    )

    assert result.claims[0].citation_ids == [2]


def test_parse_citation_response_rejects_missing_json_object():
    nodes = build_nodes()

    with pytest.raises(
        ValueError,
        match="does not contain a valid JSON object",
    ):
        nodes._parse_citation_response(
            "This response contains no JSON."
        )


def test_parse_citation_response_rejects_invalid_json():
    nodes = build_nodes()

    content = """
    {
        "claims": [
            {
                "text": "Broken JSON",
                "citation_ids": [1,]
            }
        ]
    }
    """

    with pytest.raises(
        ValueError,
        match="Failed to parse JSON",
    ):
        nodes._parse_citation_response(
            content
        )


def test_parse_citation_response_rejects_invalid_schema():
    nodes = build_nodes()

    content = """
    {
        "claims": [
            {
                "text": "Invalid citation IDs",
                "citation_ids": "not-a-list"
            }
        ]
    }
    """

    with pytest.raises(
        ValueError,
    ):
        nodes._parse_citation_response(
            content
        )


# ---------------------------------------------------------
# generate()
# ---------------------------------------------------------


def test_generate_builds_context_and_llm_payload():
    nodes = build_nodes()

    documents = [
        make_document(
            "Shayan works with React."
        ),
        make_document(
            "Shayan works with FastAPI.",
            chunk_index=1,
        ),
    ]

    citation_response = make_citation_response()

    nodes.citation_service.format_context.return_value = (
        "[1] React document\n[2] FastAPI document"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = (
        citation_response.claims
    )

    nodes.citation_service.create_citations.return_value = []

    state = {
        "tenant_id": "tenant-a",
        "question": "What technologies does Shayan use?",
        "chat_history": "Previous conversation",
        "user_memory": [
            {
                "key": "name",
                "value": "Shayan",
                "memory_type": "profile",
            }
        ],
        "documents": documents,
    }

    result = nodes.generate(state)

    nodes.citation_service.format_context.assert_called_once_with(
        documents
    )

    nodes._invoke_llm_with_fallback.assert_called_once_with(
        {
            "chat_history": "Previous conversation",
            "user_memory": state["user_memory"],
            "question": "What technologies does Shayan use?",
            "context": (
                "[1] React document\n[2] FastAPI document"
            ),
        }
    )

    assert result["answer"] == (
        "Shayan is a Full Stack Developer."
    )

    assert result["claims"] == (
        citation_response.claims
    )


def test_generate_uses_empty_defaults_for_optional_state():
    nodes = build_nodes()

    documents = [
        make_document(
            "Relevant document."
        )
    ]

    citation_response = make_citation_response()

    nodes.citation_service.format_context.return_value = (
        "Context"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = (
        citation_response.claims
    )

    nodes.citation_service.create_citations.return_value = []

    state = {
        "question": "Question",
        "documents": documents,
    }

    result = nodes.generate(state)

    nodes._invoke_llm_with_fallback.assert_called_once_with(
        {
            "chat_history": "",
            "user_memory": [],
            "question": "Question",
            "context": "Context",
        }
    )

    assert result["answer"] == (
        "Shayan is a Full Stack Developer."
    )


def test_generate_validates_citation_claims():
    nodes = build_nodes()

    documents = [
        make_document(
            "Document one."
        ),
        make_document(
            "Document two.",
            chunk_index=1,
        ),
    ]

    citation_response = make_citation_response(
        claims=[
            make_claim(
                text="Claim one.",
                citation_ids=[1, 99],
            )
        ]
    )

    validated_claim = make_claim(
        text="Claim one.",
        citation_ids=[1],
    )

    nodes.citation_service.format_context.return_value = (
        "Context"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = [
        validated_claim
    ]

    nodes.citation_service.create_citations.return_value = []

    state = {
        "question": "Question",
        "documents": documents,
    }

    result = nodes.generate(state)

    nodes.citation_validator.validate.assert_called_once_with(
        claims=citation_response.claims,
        documents=documents,
    )

    assert result["claims"] == [
        validated_claim
    ]

    assert result["answer"] == "Claim one."


def test_generate_keeps_only_citations_referenced_by_valid_claims():
    nodes = build_nodes()

    documents = [
        make_document(
            "Document one."
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

    citation_response = make_citation_response(
        claims=[
            make_claim(
                text="Claim one.",
                citation_ids=[1, 3],
            )
        ]
    )

    validated_claim = make_claim(
        text="Claim one.",
        citation_ids=[1],
    )

    citation_one = Citation(
        citation_id=1,
        file_name="resume.pdf",
        page=None,
        chunk_index=0,
        source="",
    )

    citation_two = Citation(
        citation_id=2,
        file_name="resume.pdf",
        page=None,
        chunk_index=1,
        source="",
    )

    citation_three = Citation(
        citation_id=3,
        file_name="resume.pdf",
        page=None,
        chunk_index=2,
        source="",
    )

    nodes.citation_service.format_context.return_value = (
        "Context"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = [
        validated_claim
    ]

    nodes.citation_service.create_citations.return_value = [
        citation_one,
        citation_two,
        citation_three,
    ]

    state = {
        "question": "Question",
        "documents": documents,
    }

    result = nodes.generate(state)

    assert result["citations"] == [
        citation_one
    ]


def test_generate_builds_answer_from_claim_text():
    nodes = build_nodes()

    claims = [
        make_claim(
            text="First claim.",
            citation_ids=[1],
        ),
        make_claim(
            text="Second claim.",
            citation_ids=[2],
        ),
        make_claim(
            text="Third claim.",
            citation_ids=[],
        ),
    ]

    citation_response = make_citation_response(
        claims=claims
    )

    nodes.citation_service.format_context.return_value = (
        "Context"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = claims

    nodes.citation_service.create_citations.return_value = []

    state = {
        "question": "Question",
        "documents": [],
    }

    result = nodes.generate(state)

    assert result["answer"] == (
        "First claim.\n"
        "Second claim.\n"
        "Third claim."
    )


def test_generate_preserves_existing_state():
    nodes = build_nodes()

    citation_response = make_citation_response()

    nodes.citation_service.format_context.return_value = (
        "Context"
    )

    nodes._invoke_llm_with_fallback = MagicMock(
        return_value=citation_response
    )

    nodes.citation_validator.validate.return_value = (
        citation_response.claims
    )

    nodes.citation_service.create_citations.return_value = []

    state = {
        "tenant_id": "tenant-a",
        "question": "Question",
        "chat_history": "History",
        "user_memory": [
            {
                "key": "name",
                "value": "Shayan",
                "memory_type": "profile",
            }
        ],
        "rewritten_query": "Rewritten",
        "documents": [],
    }

    result = nodes.generate(state)

    assert result["tenant_id"] == "tenant-a"
    assert result["question"] == "Question"
    assert result["chat_history"] == "History"
    assert result["user_memory"] == state["user_memory"]
    assert result["rewritten_query"] == "Rewritten"
    assert "answer" in result
    assert "claims" in result
    assert "citations" in result