
import json
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.rag.citations.models import CitationResponse
from app.rag.citations.service import CitationService
from app.rag.citations.validator import CitationValidator
from app.rag.graph.state import RAGState
from app.rag.prompts.generation import generation_prompt
from app.rag.query.rewriting import QueryRewriter
from app.rag.retrieval.hybrid import HybridRetriever

settings = get_settings()


class RAGNodes:
    """
    LangGraph nodes for the production RAG pipeline.

    Pipeline:

        Question
            ↓
        Query Rewrite
            ↓
        Hybrid Retrieval
            ↓
        Cross Encoder Reranking
            ↓
        Gemini Structured Generation
            ↓
        Retry / Backoff
            ↓
        Groq Structured Fallback
            ↓
        Citation Validation
            ↓
        Final Answer
    """

    def __init__(
        self,
        query_rewriter: QueryRewriter,
        retriever: HybridRetriever,
    ):
        self.query_rewriter = query_rewriter
        self.retriever = retriever

        self.citation_service = CitationService()
        self.citation_validator = CitationValidator()

        # ============================================================
        # PRIMARY LLM — GEMINI
        # ============================================================

        self.gemini_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_output_tokens=768,
        )

        # ============================================================
        # FALLBACK LLM — GROQ
        # ============================================================

        self.groq_llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=768,
        )

        # ============================================================
        # STRUCTURED OUTPUT CHAINS
        # ============================================================
        #
        # The LLM is instructed by LangChain to return data matching
        # CitationResponse instead of returning arbitrary JSON text.
        #
        # This prevents malformed structures such as:
        #
        #     "citation_ids": [1], [3]
        #
        # because citation_ids must be a list[int].
        # ============================================================

        self.gemini_chain = (
            generation_prompt
            | self.gemini_llm.with_structured_output(
                CitationResponse
            )
        )

        self.groq_chain = (
            generation_prompt
            | self.groq_llm.with_structured_output(
                CitationResponse
            )
        )

    # =================================================================
    # QUERY REWRITE
    # =================================================================

    def rewrite(
        self,
        state: RAGState,
    ) -> RAGState:
        rewritten_query = (
            self.query_rewriter.rewrite(
                question=state["question"],
                chat_history=state.get(
                    "chat_history",
                    "",
                ),
            )
        )

        print("\n" + "=" * 70)
        print("QUERY REWRITE")
        print("=" * 70)

        print(
            f"Original question: "
            f"{state['question']}"
        )

        print(
            f"Rewritten query: "
            f"{rewritten_query}"
        )

        print("=" * 70)

        return {
            **state,
            "rewritten_query": rewritten_query,
        }

    # =================================================================
    # HYBRID RETRIEVAL + RERANKING
    # =================================================================

    def retrieve(
        self,
        state: RAGState,
    ) -> RAGState:
        tenant_id = state.get(
            "tenant_id"
        )

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for RAG retrieval."
            )

        documents = self.retriever.search(
            query=state["question"],
            chat_history=state.get(
                "chat_history",
                "",
            ),
            rewritten_query=state.get(
                "rewritten_query",
            ),
            dense_k=10,
            bm25_k=10,
            top_k=3,
            tenant_id=tenant_id,
        )

        print("\n" + "=" * 70)
        print("FINAL RETRIEVED DOCUMENTS")
        print("=" * 70)

        print(
            f"Tenant: {tenant_id}"
        )

        print(
            f"Documents passed to generation: "
            f"{len(documents)}"
        )

        print("-" * 70)

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            print(
                f"\nSOURCE [{index}]"
            )

            print(
                f"File: "
                f"{metadata.get('file_name', 'Unknown')}"
            )

            print(
                f"Chunk: "
                f"{metadata.get('chunk_index', 'Unknown')}"
            )

            print(
                f"Tenant: "
                f"{metadata.get('tenant_id', 'Unknown')}"
            )

            print(
                f"Rerank score: "
                f"{metadata.get('rerank_score', 'N/A')}"
            )

            print("Content:")

            print(
                document.page_content
            )

        print("\n" + "=" * 70)

        return {
            **state,
            "documents": documents,
        }

    # =================================================================
    # RESPONSE TEXT NORMALIZATION
    # =================================================================

    def _extract_response_text(
        self,
        content: Any,
    ) -> str:
        """
        Normalize LLM response content into plain text.

        Supports:
        - str
        - list of blocks
        - dict blocks
        - objects containing .text
        """

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        if isinstance(
            content,
            list,
        ):
            text_parts: list[str] = []

            for block in content:
                if isinstance(
                    block,
                    dict,
                ):
                    block_type = block.get(
                        "type"
                    )

                    if block_type == "text":
                        text = block.get(
                            "text",
                            "",
                        )

                        if isinstance(
                            text,
                            str,
                        ):
                            text_parts.append(
                                text
                            )

                    elif "text" in block:
                        text = block.get(
                            "text",
                            "",
                        )

                        if isinstance(
                            text,
                            str,
                        ):
                            text_parts.append(
                                text
                            )

                else:
                    text = getattr(
                        block,
                        "text",
                        None,
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        text_parts.append(
                            text
                        )

            return "\n".join(
                text_parts
            ).strip()

        return str(
            content
        ).strip()

    # =================================================================
    # RETRYABLE ERROR DETECTION
    # =================================================================

    @staticmethod
    def _is_retryable_llm_error(
        exc: Exception,
    ) -> bool:
        """
        Determine whether an LLM failure is temporary
        and should trigger retry/fallback.

        Retryable examples:

        - HTTP 429
        - HTTP 503
        - HTTP 500
        - HTTP 502
        - HTTP 504
        - unavailable
        - resource exhausted
        - timeout
        - deadline exceeded
        - temporary service failures
        """

        error_text = str(
            exc
        ).lower()

        retryable_keywords = [
            "429",
            "503",
            "500",
            "502",
            "504",
            "unavailable",
            "resource exhausted",
            "rate limit",
            "rate_limit",
            "too many requests",
            "timeout",
            "timed out",
            "deadline",
            "temporarily unavailable",
            "temporary",
            "service unavailable",
        ]

        return any(
            keyword in error_text
            for keyword in retryable_keywords
        )

    # =================================================================
    # GEMINI → RETRY → GROQ FALLBACK
    # =================================================================

    def _invoke_llm_with_fallback(
        self,
        payload: dict,
    ) -> CitationResponse:
        """
        Production-style structured LLM resilience.

        Flow:

            Gemini structured output
                ↓
            failure?
                ↓
            retry 1
                ↓
            retry 2
                ↓
            retry 3
                ↓
            Groq structured fallback

        Backoff:

            1 second
            2 seconds
            4 seconds

        Returns:
            CitationResponse
        """

        max_retries = 3
        base_delay = 1

        # ============================================================
        # GEMINI PRIMARY
        # ============================================================

        for attempt in range(
            1,
            max_retries + 1,
        ):
            try:
                print("\n" + "=" * 70)
                print("PRIMARY LLM")
                print("=" * 70)

                print(
                    "Provider: Gemini"
                )

                print(
                    f"Model: "
                    f"{settings.gemini_model}"
                )

                print(
                    f"Attempt: "
                    f"{attempt}/{max_retries}"
                )

                print("=" * 70)

                response = (
                    self.gemini_chain.invoke(
                        payload
                    )
                )

                if not isinstance(
                    response,
                    CitationResponse,
                ):
                    raise ValueError(
                        "Gemini returned an unexpected "
                        "structured output type."
                    )

                print(
                    "[LLM] Gemini structured "
                    "generation successful."
                )

                return response

            except Exception as exc:
                print("\n" + "=" * 70)
                print("GEMINI ERROR")
                print("=" * 70)

                print(
                    f"Attempt: "
                    f"{attempt}/{max_retries}"
                )

                print(
                    f"Error: "
                    f"{exc}"
                )

                print("=" * 70)

                # ----------------------------------------------------
                # Non-retryable error
                # ----------------------------------------------------

                if not self._is_retryable_llm_error(
                    exc
                ):
                    print(
                        "[LLM] Non-retryable Gemini "
                        "error detected."
                    )

                    raise

                # ----------------------------------------------------
                # Retries exhausted
                # ----------------------------------------------------

                if attempt >= max_retries:
                    print(
                        "[LLM] Gemini retries "
                        "exhausted."
                    )

                    break

                # ----------------------------------------------------
                # Exponential backoff
                # ----------------------------------------------------

                delay = (
                    base_delay
                    * (2 ** (attempt - 1))
                )

                print(
                    "[LLM] Gemini temporary "
                    "failure."
                )

                print(
                    f"[LLM] Retrying in "
                    f"{delay} second(s)..."
                )

                time.sleep(
                    delay
                )

        # ============================================================
        # GROQ FALLBACK
        # ============================================================

        print("\n" + "=" * 70)
        print("LLM FALLBACK")
        print("=" * 70)

        print(
            "Primary provider Gemini failed."
        )

        print(
            "Switching to Groq fallback."
        )

        print(
            "Provider: Groq"
        )

        print(
            f"Model: "
            f"{settings.groq_model}"
        )

        print("=" * 70)

        try:
            response = (
                self.groq_chain.invoke(
                    payload
                )
            )

            if not isinstance(
                response,
                CitationResponse,
            ):
                raise ValueError(
                    "Groq returned an unexpected "
                    "structured output type."
                )

            print(
                "[LLM] Groq structured fallback "
                "generation successful."
            )

            return response

        except Exception as exc:
            print("\n" + "=" * 70)
            print("GROQ FALLBACK ERROR")
            print("=" * 70)

            print(
                f"Error: "
                f"{exc}"
            )

            print("=" * 70)

            raise RuntimeError(
                "Both Gemini and Groq "
                "structured generation failed."
            ) from exc

    # =================================================================
    # CITATION JSON PARSER
    # =================================================================

    def _parse_citation_response(
        self,
        content: str,
    ) -> CitationResponse:
        """
        Defensive parser for raw LLM citation JSON.

        The primary generation path now uses structured output
        directly and does not require this parser.

        This method is retained as a defensive utility for cases
        where raw JSON text may need to be parsed explicitly.

        Handles:

        1. Pure JSON
        2. JSON wrapped in ```json
        3. JSON embedded inside additional text
        """

        content = content.strip()

        # ------------------------------------------------------------
        # Remove markdown code fence
        # ------------------------------------------------------------

        if content.startswith(
            "```"
        ):
            lines = content.splitlines()

            if (
                lines
                and lines[0].startswith(
                    "```"
                )
            ):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):
                lines = lines[:-1]

            content = "\n".join(
                lines
            ).strip()

        # ------------------------------------------------------------
        # Attempt direct JSON parsing
        # ------------------------------------------------------------

        try:
            data = json.loads(
                content
            )

            return CitationResponse.model_validate(
                data
            )

        except (
            json.JSONDecodeError,
            ValueError,
        ):
            pass

        # ------------------------------------------------------------
        # Attempt extracting JSON object
        # ------------------------------------------------------------

        start = content.find(
            "{"
        )

        end = content.rfind(
            "}"
        )

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            raise ValueError(
                "LLM response does not contain "
                "a valid JSON object."
            )

        json_content = content[
            start : end + 1
        ]

        try:
            data = json.loads(
                json_content
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON from "
                f"LLM response: {exc}"
            ) from exc

        return CitationResponse.model_validate(
            data
        )

    # =================================================================
    # GENERATION
    # =================================================================

    def generate(
        self,
        state: RAGState,
    ) -> RAGState:
        """
        Generate answer from retrieved documents.

        LLM flow:

            Gemini structured output
                ↓
            Retry / Backoff
                ↓
            Groq structured fallback
                ↓
            Citation validation
                ↓
            Final answer
        """

        documents = state.get(
            "documents",
            [],
        )

        # ============================================================
        # BUILD CONTEXT
        # ============================================================

        context = (
            self.citation_service.format_context(
                documents
            )
        )

        print("\n" + "=" * 70)
        print("GENERATION CONTEXT")
        print("=" * 70)

        print(
            context
        )

        print("=" * 70)

        # ============================================================
        # DURABLE USER MEMORY
        # ============================================================

        user_memory = state.get(
            "user_memory",
            [],
        )

        print("\n" + "=" * 70)
        print("DURABLE USER MEMORY")
        print("=" * 70)

        if user_memory:
            for memory in user_memory:
                print(
                    f"{memory.get('key')}: "
                    f"{memory.get('value')}"
                )
        else:
            print(
                "No durable user memory found."
            )

        print("=" * 70)

        # ============================================================
        # GENERATION PAYLOAD
        # ============================================================

        payload = {
            "chat_history": state.get(
                "chat_history",
                "",
            ),
            "user_memory": user_memory,
            "question": state["question"],
            "context": context,
        }

        # ============================================================
        # STRUCTURED LLM GENERATION
        # ============================================================

        citation_response = (
            self._invoke_llm_with_fallback(
                payload
            )
        )

        print("\n" + "=" * 70)
        print("STRUCTURED LLM RESPONSE")
        print("=" * 70)

        print(
            citation_response.model_dump_json(
                indent=2
            )
        )

        print("=" * 70)

        # ============================================================
        # PARSED LLM CLAIMS
        # ============================================================

        print("\n" + "=" * 70)
        print("PARSED LLM CLAIMS")
        print("=" * 70)

        print(
            f"Number of claims: "
            f"{len(citation_response.claims)}"
        )

        for index, claim in enumerate(
            citation_response.claims,
            start=1,
        ):
            print(
                f"\nClaim {index}:"
            )

            print(
                f"Text: "
                f"{claim.text}"
            )

            print(
                f"Citation IDs: "
                f"{claim.citation_ids}"
            )

        print("=" * 70)

        # ============================================================
        # CITATION VALIDATION
        # ============================================================

        claims = (
            self.citation_validator.validate(
                claims=citation_response.claims,
                documents=documents,
            )
        )

        # ============================================================
        # VALID CITATION IDS
        # ============================================================

        valid_citation_ids = {
            citation_id
            for claim in claims
            for citation_id in claim.citation_ids
        }

        # ============================================================
        # CREATE CITATIONS
        # ============================================================

        citations = (
            self.citation_service.create_citations(
                documents
            )
        )

        citations = [
            citation
            for citation in citations
            if citation.citation_id
            in valid_citation_ids
        ]

        # ============================================================
        # FINAL ANSWER
        # ============================================================

        answer = "\n".join(
            claim.text
            for claim in claims
        )

        print("\n" + "=" * 70)
        print("FINAL GENERATION RESULT")
        print("=" * 70)

        print(
            f"Answer: "
            f"{answer}"
        )

        print(
            f"Citations: "
            f"{len(citations)}"
        )

        print("=" * 70)

        # ============================================================
        # RETURN UPDATED STATE
        # ============================================================

        return {
            **state,
            "answer": answer,
            "claims": claims,
            "citations": citations,
        }
