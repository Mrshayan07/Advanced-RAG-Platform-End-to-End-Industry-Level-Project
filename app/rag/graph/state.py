from typing import TypedDict

from langchain_core.documents import Document

from app.rag.citations.models import Citation, CitationClaim


class RAGState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph
    RAG workflow.
    """

    # ============================================================
    # SECURITY CONTEXT
    # ============================================================

    # Authenticated tenant identifier.
    #
    # This value originates from the verified JWT and is passed
    # through LangGraph to tenant-aware retrieval.
    tenant_id: str

    # ============================================================
    # USER QUESTION
    # ============================================================

    # Original user question
    question: str

    # ============================================================
    # SHORT-TERM MEMORY
    # ============================================================

    # Short-term conversation history from Redis
    chat_history: str

    # ============================================================
    # DURABLE USER MEMORY
    # ============================================================

    # Durable user-specific memory from PostgreSQL
    #
    # Example:
    #
    # [
    #     {
    #         "key": "name",
    #         "value": "Shayan Javed",
    #         "memory_type": "profile",
    #     },
    #     {
    #         "key": "university",
    #         "value": "Sindh Madressatul Islam University",
    #         "memory_type": "education",
    #     },
    # ]
    user_memory: list[dict[str, str]]

    # ============================================================
    # QUERY REWRITING
    # ============================================================

    # Query after query rewriting
    rewritten_query: str

    # ============================================================
    # RETRIEVAL
    # ============================================================

    # Documents returned by hybrid retrieval + reranking
    documents: list[Document]

    # ============================================================
    # GENERATION
    # ============================================================

    # Final generated answer
    answer: str

    # ============================================================
    # CITATIONS
    # ============================================================

    # Structured answer claims
    claims: list[CitationClaim]

    # Citations attached to the final answer
    citations: list[Citation]