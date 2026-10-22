from __future__ import annotations

from langchain_core.documents import Document
from langsmith import traceable

from app.rag.embeddings.service import EmbeddingService
from app.rag.query.rewriting import QueryRewriter
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.bm25 import BM25Service
from app.rag.retrieval.fusion import RRFFusion
from app.rag.vectorstore.qdrant import QdrantService


class HybridRetriever:
    """
    Production-oriented tenant-aware hybrid retriever.

    Retrieval pipeline:

        User Question
              ↓
        Query Rewriting
              ↓
        ┌───────────────────────────────┐
        ↓                               ↓
    Dense Retrieval                BM25 Retrieval
    Qdrant + tenant filter        Tenant-specific index
        │                               │
        └────────────────┬──────────────┘
                         ↓
                    RRF Fusion
                         ↓
              Reranker Candidate Cap
                         ↓
              Cross-Encoder Rerank
                         ↓
                      Top-K

    Tenant isolation:

        authenticated tenant_id
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
    Qdrant filter      BM25 index
        ↓                 ↓
    Tenant docs       Tenant docs
        └────────┬────────┘
                 ↓
             RRF Fusion
                 ↓
       Reranker Candidates
                 ↓
          Cross-Encoder
                 ↓
          Final documents

    Security invariant:

        Every retrieval path MUST receive tenant_id.

    This prevents cross-tenant document leakage through
    either dense retrieval or lexical BM25 retrieval.
    """

    def __init__(
        self,
        bm25_service: BM25Service,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        reranker: CrossEncoderReranker,
        query_rewriter: QueryRewriter,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        enable_logging: bool = True,
        rerank_candidate_k: int = 5,
    ):
        self.bm25 = bm25_service
        self.embeddings = embedding_service
        self.qdrant = qdrant_service

        self.fusion = RRFFusion()

        self.reranker = reranker
        self.query_rewriter = query_rewriter

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        self.enable_logging = enable_logging

        if rerank_candidate_k <= 0:
            raise ValueError(
                "rerank_candidate_k must be greater than zero."
            )

        self.rerank_candidate_k = rerank_candidate_k

    # =========================================================
    # LOGGING
    # =========================================================

    def _log(
        self,
        stage: str,
        message: str,
    ) -> None:
        if self.enable_logging:
            print(
                f"\n[HybridRetriever] "
                f"{stage}: {message}"
            )

    # =========================================================
    # LANGSMITH: DENSE RETRIEVAL
    # =========================================================

    @traceable(
        name="Dense Retrieval",
        run_type="retriever",
    )
    def _dense_retrieve(
        self,
        rewritten_query: str,
        dense_k: int,
        tenant_id: str,
    ) -> list[Document]:
        """
        Execute tenant-aware dense vector retrieval.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for dense retrieval."
            )

        self._log(
            "DENSE_RETRIEVAL",
            (
                f"Top-K: {dense_k} | "
                f"Tenant: {tenant_id}"
            ),
        )

        dense_documents: list[Document] = []

        query_vector = self.embeddings.embed_query(
            rewritten_query
        )

        dense_results = self.qdrant.search(
            query_vector,
            limit=dense_k,
            tenant_id=tenant_id,
        )

        for result in dense_results:
            payload = result.payload or {}

            text = payload.get(
                "text",
                "",
            )

            if not text:
                continue

            metadata = {
                key: value
                for key, value in payload.items()
                if key != "text"
            }

            dense_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        # Defensive post-retrieval tenant validation.
        for document in dense_documents:
            document_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if str(document_tenant_id) != str(tenant_id):
                raise RuntimeError(
                    "Dense retrieval tenant isolation violation "
                    "detected. "
                    f"Expected tenant_id={tenant_id}, "
                    f"got tenant_id={document_tenant_id}."
                )

        self._log(
            "DENSE_RETRIEVAL",
            (
                f"Retrieved: "
                f"{len(dense_documents)} documents | "
                f"Tenant: {tenant_id}"
            ),
        )

        print("\n" + "=" * 70)
        print("DENSE RETRIEVAL DOCUMENTS")
        print("=" * 70)

        for index, document in enumerate(
            dense_documents,
            start=1,
        ):
            metadata = document.metadata

            content_preview = (
                document.page_content
                .replace("\n", " ")
                [:200]
            )

            print(
                f"Rank {index}: "
                f"file="
                f"{metadata.get('file_name', 'Unknown')} | "
                f"chunk="
                f"{metadata.get('chunk_index', 'Unknown')} | "
                f"tenant="
                f"{metadata.get('tenant_id', 'Unknown')}"
            )

            print(
                f"Content: {content_preview}"
            )

            print("-" * 70)

        print("=" * 70)

        return dense_documents

    # =========================================================
    # LANGSMITH: BM25 RETRIEVAL
    # =========================================================

    @traceable(
        name="BM25 Retrieval",
        run_type="retriever",
    )
    def _bm25_retrieve(
        self,
        rewritten_query: str,
        bm25_k: int,
        tenant_id: str,
    ) -> list[Document]:
        """
        Execute tenant-aware lexical BM25 retrieval.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for BM25 retrieval."
            )

        self._log(
            "BM25_RETRIEVAL",
            (
                f"Top-K: {bm25_k} | "
                f"Tenant: {tenant_id}"
            ),
        )

        bm25_documents = self.bm25.search(
            rewritten_query,
            tenant_id=tenant_id,
            k=bm25_k,
        )

        # Defensive post-retrieval tenant validation.
        for document in bm25_documents:
            document_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if str(document_tenant_id) != str(tenant_id):
                raise RuntimeError(
                    "BM25 retrieval tenant isolation violation "
                    "detected. "
                    f"Expected tenant_id={tenant_id}, "
                    f"got tenant_id={document_tenant_id}."
                )

        self._log(
            "BM25_RETRIEVAL",
            (
                f"Retrieved: "
                f"{len(bm25_documents)} documents | "
                f"Tenant: {tenant_id}"
            ),
        )

        print("\n" + "=" * 70)
        print("BM25 RETRIEVAL DOCUMENTS")
        print("=" * 70)

        for index, document in enumerate(
            bm25_documents,
            start=1,
        ):
            metadata = document.metadata

            content_preview = (
                document.page_content
                .replace("\n", " ")
                [:200]
            )

            print(
                f"Rank {index}: "
                f"file="
                f"{metadata.get('file_name', 'Unknown')} | "
                f"chunk="
                f"{metadata.get('chunk_index', 'Unknown')} | "
                f"tenant="
                f"{metadata.get('tenant_id', 'Unknown')}"
            )

            print(
                f"Content: {content_preview}"
            )

            print("-" * 70)

        print("=" * 70)

        return bm25_documents

    # =========================================================
    # LANGSMITH: RRF FUSION
    # =========================================================

    @traceable(
        name="RRF Fusion",
        run_type="chain",
    )
    def _fuse_results(
        self,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        fusion_top_k: int,
        tenant_id: str,
    ) -> list[Document]:
        """
        Fuse dense and BM25 results using Reciprocal Rank
        Fusion.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for RRF fusion."
            )

        self._log(
            "FUSION",
            (
                f"Fusing dense "
                f"({len(dense_documents)}) + "
                f"bm25 "
                f"({len(bm25_documents)}) results | "
                f"Tenant: {tenant_id}"
            ),
        )

        fused_documents = self.fusion.fuse(
            result_lists=[
                dense_documents,
                bm25_documents,
            ],
            top_k=fusion_top_k,
        )

        # Final defense-in-depth tenant validation.
        for document in fused_documents:
            document_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if str(document_tenant_id) != str(tenant_id):
                raise RuntimeError(
                    "RRF fusion tenant isolation violation "
                    "detected. "
                    f"Expected tenant_id={tenant_id}, "
                    f"got tenant_id={document_tenant_id}."
                )

        self._log(
            "FUSION",
            (
                f"Fused: "
                f"{len(fused_documents)} documents | "
                f"Tenant: {tenant_id}"
            ),
        )

        print("\n" + "=" * 70)
        print("RRF FUSED DOCUMENTS")
        print("=" * 70)

        for index, document in enumerate(
            fused_documents,
            start=1,
        ):
            metadata = document.metadata

            content_preview = (
                document.page_content
                .replace("\n", " ")
                [:200]
            )

            print(
                f"Rank {index}: "
                f"file="
                f"{metadata.get('file_name', 'Unknown')} | "
                f"chunk="
                f"{metadata.get('chunk_index', 'Unknown')} | "
                f"tenant="
                f"{metadata.get('tenant_id', 'Unknown')}"
            )

            print(
                f"Content: {content_preview}"
            )

            print("-" * 70)

        print("=" * 70)

        return fused_documents

    # =========================================================
    # DEFENSIVE TENANT VALIDATION
    # =========================================================

    def _validate_tenant_documents(
        self,
        documents: list[Document],
        tenant_id: str,
        stage: str,
    ) -> None:
        """
        Verify that every document belongs to the
        authenticated tenant.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for document validation."
            )

        for document in documents:
            document_tenant_id = document.metadata.get(
                "tenant_id"
            )

            if str(document_tenant_id) != str(tenant_id):
                raise RuntimeError(
                    f"Tenant isolation violation at {stage}. "
                    f"Expected tenant_id={tenant_id}, "
                    f"got tenant_id={document_tenant_id}."
                )

    # =========================================================
    # RERANKER CANDIDATE SELECTION
    # =========================================================

    def _select_rerank_candidates(
        self,
        fused_documents: list[Document],
    ) -> list[Document]:
        """
        Limit the number of documents sent to the expensive
        cross-encoder reranker.

        RRF still performs its normal fusion first.

        Only the expensive reranking stage is capped.

        Example:

            RRF output:        7 documents
            Candidate limit:  5 documents
            Cross-Encoder:    5 documents
            Final output:     3 documents
        """

        candidates = fused_documents[
            : self.rerank_candidate_k
        ]

        self._log(
            "RERANK_CANDIDATES",
            (
                f"Selected {len(candidates)} of "
                f"{len(fused_documents)} fused documents "
                f"for Cross-Encoder | "
                f"Candidate limit: "
                f"{self.rerank_candidate_k}"
            ),
        )

        return candidates

    # =========================================================
    # MAIN SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        chat_history: str = "",
        rewritten_query: str | None = None,
        dense_k: int = 10,
        bm25_k: int = 10,
        top_k: int = 5,
        tenant_id: str | None = None,
    ) -> list[Document]:
        """
        Execute complete tenant-aware hybrid retrieval.

        Steps:

        1. Query rewriting
        2. Tenant-aware dense retrieval
        3. Tenant-aware BM25 retrieval
        4. RRF fusion
        5. Reranker candidate selection
        6. Cross-encoder reranking
        7. Final tenant validation
        8. Top-K documents

        tenant_id is mandatory.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for hybrid retrieval."
            )

        normalized_tenant_id = str(tenant_id)

        # =====================================================
        # 1. QUERY REWRITING
        # =====================================================

        self._log(
            "QUERY_REWRITING",
            f"Original: {query}",
        )

        if rewritten_query is None:
            rewritten_query = (
                self.query_rewriter.rewrite(
                    question=query,
                    chat_history=chat_history,
                )
            )

        self._log(
            "QUERY_REWRITING",
            f"Rewritten: {rewritten_query}",
        )

        # =====================================================
        # 2. DENSE RETRIEVAL
        # =====================================================

        try:
            dense_documents = self._dense_retrieve(
                rewritten_query=rewritten_query,
                dense_k=dense_k,
                tenant_id=normalized_tenant_id,
            )

            self._validate_tenant_documents(
                documents=dense_documents,
                tenant_id=normalized_tenant_id,
                stage="DENSE_RETRIEVAL",
            )

        except Exception as exc:
            self._log(
                "DENSE_RETRIEVAL",
                f"ERROR: {exc}",
            )

            dense_documents = []

        # =====================================================
        # 3. BM25 RETRIEVAL
        # =====================================================

        try:
            bm25_documents = self._bm25_retrieve(
                rewritten_query=rewritten_query,
                bm25_k=bm25_k,
                tenant_id=normalized_tenant_id,
            )

            self._validate_tenant_documents(
                documents=bm25_documents,
                tenant_id=normalized_tenant_id,
                stage="BM25_RETRIEVAL",
            )

        except Exception as exc:
            self._log(
                "BM25_RETRIEVAL",
                f"ERROR: {exc}",
            )

            bm25_documents = []

        # =====================================================
        # 4. RRF FUSION
        # =====================================================

        fused_documents = self._fuse_results(
            dense_documents=dense_documents,
            bm25_documents=bm25_documents,
            fusion_top_k=max(
                dense_k,
                bm25_k,
            ),
            tenant_id=normalized_tenant_id,
        )

        self._validate_tenant_documents(
            documents=fused_documents,
            tenant_id=normalized_tenant_id,
            stage="RRF_FUSION",
        )

        # =====================================================
        # 5. RERANKER CANDIDATE SELECTION
        # =====================================================

        rerank_candidates = self._select_rerank_candidates(
            fused_documents=fused_documents,
        )

        self._validate_tenant_documents(
            documents=rerank_candidates,
            tenant_id=normalized_tenant_id,
            stage="RERANK_CANDIDATES",
        )

        # =====================================================
        # 6. CROSS-ENCODER RERANKING
        # =====================================================

        self._log(
            "RERANKING",
            (
                f"Reranking "
                f"{len(rerank_candidates)} candidates "
                f"(returning top {top_k}) | "
                f"Tenant: {normalized_tenant_id}"
            ),
        )

        try:
            reranked_documents = (
                self.reranker.rerank(
                    query=rewritten_query,
                    documents=rerank_candidates,
                    top_k=top_k,
                )
            )

            self._validate_tenant_documents(
                documents=reranked_documents,
                tenant_id=normalized_tenant_id,
                stage="RERANKING",
            )

            self._log(
                "RERANKING",
                (
                    f"Final: "
                    f"{len(reranked_documents)} "
                    f"documents returned | "
                    f"Tenant: {normalized_tenant_id}"
                ),
            )

            if reranked_documents:
                top_score = (
                    reranked_documents[0]
                    .metadata
                    .get(
                        "rerank_score",
                        "N/A",
                    )
                )

                self._log(
                    "RERANKING",
                    f"Top score: {top_score}",
                )

        except Exception as exc:
            self._log(
                "RERANKING",
                f"ERROR: {exc}",
            )

            reranked_documents = (
                rerank_candidates[:top_k]
            )

            self._validate_tenant_documents(
                documents=reranked_documents,
                tenant_id=normalized_tenant_id,
                stage="RERANKING_FALLBACK",
            )

            self._log(
                "RERANKING_FALLBACK",
                (
                    f"Returned "
                    f"{len(reranked_documents)} "
                    f"documents from reranker "
                    f"candidate set | "
                    f"Tenant: {normalized_tenant_id}"
                ),
            )

        return reranked_documents

    # =========================================================
    # SEARCH WITH SCORES
    # =========================================================

    def search_with_scores(
        self,
        query: str,
        chat_history: str = "",
        top_k: int = 5,
        tenant_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Execute tenant-aware retrieval and return documents
        together with their reranker scores.

        Useful for:

        - Evaluation
        - Debugging
        - Retrieval analytics
        - Threshold calibration

        tenant_id is mandatory.
        """

        if not tenant_id:
            raise ValueError(
                "tenant_id is required for hybrid retrieval."
            )

        normalized_tenant_id = str(tenant_id)

        # =====================================================
        # 1. QUERY REWRITING
        # =====================================================

        rewritten_query = (
            self.query_rewriter.rewrite(
                question=query,
                chat_history=chat_history,
            )
        )

        # =====================================================
        # 2. DENSE RETRIEVAL
        # =====================================================

        dense_documents = self._dense_retrieve(
            rewritten_query=rewritten_query,
            dense_k=10,
            tenant_id=normalized_tenant_id,
        )

        self._validate_tenant_documents(
            documents=dense_documents,
            tenant_id=normalized_tenant_id,
            stage="DENSE_RETRIEVAL",
        )

        # =====================================================
        # 3. BM25 RETRIEVAL
        # =====================================================

        bm25_documents = self._bm25_retrieve(
            rewritten_query=rewritten_query,
            bm25_k=10,
            tenant_id=normalized_tenant_id,
        )

        self._validate_tenant_documents(
            documents=bm25_documents,
            tenant_id=normalized_tenant_id,
            stage="BM25_RETRIEVAL",
        )

        # =====================================================
        # 4. RRF FUSION
        # =====================================================

        fused_documents = self._fuse_results(
            dense_documents=dense_documents,
            bm25_documents=bm25_documents,
            fusion_top_k=10,
            tenant_id=normalized_tenant_id,
        )

        self._validate_tenant_documents(
            documents=fused_documents,
            tenant_id=normalized_tenant_id,
            stage="RRF_FUSION",
        )

        # =====================================================
        # 5. RERANKER CANDIDATE SELECTION
        # =====================================================

        rerank_candidates = self._select_rerank_candidates(
            fused_documents=fused_documents,
        )

        self._validate_tenant_documents(
            documents=rerank_candidates,
            tenant_id=normalized_tenant_id,
            stage="RERANK_CANDIDATES",
        )

        # =====================================================
        # 6. CROSS-ENCODER RERANKING
        # =====================================================

        reranked_documents = (
            self.reranker.rerank(
                query=rewritten_query,
                documents=rerank_candidates,
                top_k=top_k,
            )
        )

        self._validate_tenant_documents(
            documents=reranked_documents,
            tenant_id=normalized_tenant_id,
            stage="RERANKING",
        )

        # =====================================================
        # 7. RETURN DOCUMENT + SCORE
        # =====================================================

        scored_results: list[
            tuple[Document, float]
        ] = []

        for document in reranked_documents:
            score = float(
                document.metadata.get(
                    "rerank_score",
                    0.0,
                )
            )

            scored_results.append(
                (
                    document,
                    score,
                )
            )

        return scored_results