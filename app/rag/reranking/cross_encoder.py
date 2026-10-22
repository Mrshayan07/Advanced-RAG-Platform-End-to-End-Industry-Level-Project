
from __future__ import annotations

from time import perf_counter

from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        min_score_ratio: float = 0.0,
    ):
        print("\n" + "=" * 70)
        print("INITIALIZING CROSS-ENCODER RERANKER")
        print("=" * 70)
        print(f"Model: {model_name}")

        start = perf_counter()
        self.model = CrossEncoder(model_name)
        elapsed = perf_counter() - start

        print(f"Model loaded in: {elapsed:.3f}s")
        print("=" * 70)

        self.model_name = model_name
        self.min_score_ratio = min_score_ratio

    @staticmethod
    def _to_float(score) -> float:
        try:
            return float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid cross-encoder score: {score!r}"
            ) from exc

    def _diagnose_input_lengths(
        self,
        query: str,
        documents: list[Document],
    ) -> None:
        """
        Diagnostic only.

        Measures the actual tokenized length of every
        query-document pair without changing reranker behavior.
        """

        tokenizer = self.model.tokenizer

        start = perf_counter()

        try:
            encoded = tokenizer(
                [query] * len(documents),
                [document.page_content for document in documents],
                truncation=False,
                padding=False,
                add_special_tokens=True,
                return_length=True,
            )

            lengths = encoded["length"]

        except Exception as exc:
            elapsed = perf_counter() - start
            print(
                "[CrossEncoder] Token-length diagnostic failed: "
                f"{exc}"
            )
            print(
                "[CrossEncoder] Diagnostic elapsed: "
                f"{elapsed:.3f}s"
            )
            return

        elapsed = perf_counter() - start

        print("\n" + "-" * 70)
        print("CROSS-ENCODER INPUT LENGTH DIAGNOSTICS")
        print("-" * 70)
        print(
            f"Tokenizer: "
            f"{type(tokenizer).__name__}"
        )
        print(
            f"Tokenizer max length: "
            f"{tokenizer.model_max_length}"
        )
        print(
            f"Pair special tokens: "
            f"{tokenizer.num_special_tokens_to_add(pair=True)}"
        )
        print(
            f"Tokenization diagnostic time: "
            f"{elapsed:.4f}s"
        )
        print(
            f"Documents received: "
            f"{len(documents)}"
        )

        if lengths:
            min_length = min(lengths)
            max_length = max(lengths)
            avg_length = sum(lengths) / len(lengths)

            print(f"Min pair tokens: {min_length}")
            print(f"Max pair tokens: {max_length}")
            print(f"Avg pair tokens: {avg_length:.1f}")

        print("\nPer-document lengths:")

        for index, (document, token_length) in enumerate(
            zip(documents, lengths),
            start=1,
        ):
            chunk_id = document.metadata.get(
                "chunk_id",
                document.metadata.get("chunk_index", "N/A"),
            )

            print(
                f"  [{index}] "
                f"chunk={chunk_id} | "
                f"chars={len(document.page_content)} | "
                f"tokens={token_length}"
            )

        print("-" * 70 + "\n")

    @traceable(
        name="Cross-Encoder Reranking",
        run_type="chain",
    )
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:

        total_start = perf_counter()

        if not documents:
            print("[CrossEncoder] No documents received.")
            return []

        if top_k <= 0:
            print(
                "[CrossEncoder] top_k <= 0. "
                "Returning no documents."
            )
            return []

        pairs_start = perf_counter()

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        pairs_elapsed = perf_counter() - pairs_start

        print(
            f"[CrossEncoder] Pair preparation: "
            f"{pairs_elapsed:.4f}s"
        )

        # ---------------------------------------------------------
        # DIAGNOSTIC ONLY
        # ---------------------------------------------------------
        # This does NOT modify the actual pairs sent to predict().
        # It only tells us how many tokens each pair contains.
        self._diagnose_input_lengths(
            query=query,
            documents=documents,
        )

        # ---------------------------------------------------------
        # ACTUAL MODEL INFERENCE
        # ---------------------------------------------------------

        inference_start = perf_counter()

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        inference_elapsed = perf_counter() - inference_start

        print(
            f"[CrossEncoder] Inference: "
            f"{inference_elapsed:.4f}s"
        )

        # ---------------------------------------------------------
        # SCORE VALIDATION
        # ---------------------------------------------------------

        validation_start = perf_counter()

        if len(scores) != len(documents):
            raise RuntimeError(
                "Cross-encoder returned a different "
                "number of scores than documents."
            )

        scored_documents = []

        for document, raw_score in zip(documents, scores):
            score = self._to_float(raw_score)
            scored_documents.append(
                (document, score)
            )

        validation_elapsed = perf_counter() - validation_start

        # ---------------------------------------------------------
        # SORTING
        # ---------------------------------------------------------

        sorting_start = perf_counter()

        ranked_documents = sorted(
            scored_documents,
            key=lambda item: item[1],
            reverse=True,
        )

        sorting_elapsed = perf_counter() - sorting_start

        if not ranked_documents:
            return []

        best_score = ranked_documents[0][1]
        worst_score = ranked_documents[-1][1]
        score_range = best_score - worst_score

        # ---------------------------------------------------------
        # FILTERING
        # ---------------------------------------------------------

        filtering_start = perf_counter()

        filtered_documents = []

        for document, score in ranked_documents:
            if best_score != 0:
                score_ratio = score / best_score
            else:
                score_ratio = 1.0

            document.metadata["rerank_score"] = score
            document.metadata["rerank_score_ratio"] = score_ratio
            document.metadata["reranker_model"] = self.model_name

            if (
                self.min_score_ratio <= 0
                or score_ratio >= self.min_score_ratio
            ):
                filtered_documents.append(document)

        final_documents = filtered_documents[:top_k]

        filtering_elapsed = perf_counter() - filtering_start
        total_elapsed = perf_counter() - total_start

        # ---------------------------------------------------------
        # DIAGNOSTICS
        # ---------------------------------------------------------

        print("\n" + "=" * 70)
        print("CROSS-ENCODER RERANKING DIAGNOSTICS")
        print("=" * 70)

        print(
            f"Documents received: {len(documents)}"
        )
        print(
            f"Documents returned: {len(final_documents)}"
        )
        print(
            f"Pairs preparation: {pairs_elapsed:.4f}s"
        )
        print(
            f"Inference: {inference_elapsed:.4f}s"
        )
        print(
            f"Validation: {validation_elapsed:.4f}s"
        )
        print(
            f"Sorting: {sorting_elapsed:.4f}s"
        )
        print(
            f"Filtering: {filtering_elapsed:.4f}s"
        )
        print(
            f"TOTAL RERANK TIME: {total_elapsed:.4f}s"
        )

        print(
            f"Best score: {best_score:.6f}"
        )
        print(
            f"Worst score: {worst_score:.6f}"
        )
        print(
            f"Score range: {score_range:.6f}"
        )

        print("\nRanked documents:")

        for rank, (document, score) in enumerate(
            ranked_documents,
            start=1,
        ):
            status = (
                "KEEP"
                if document in final_documents
                else "DROP"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                document.metadata.get("chunk_index", "N/A"),
            )

            print(
                f"Rank {rank}: "
                f"chunk={chunk_id} | "
                f"score={score:.6f} | "
                f"{status}"
            )

        print("=" * 70 + "\n")

        return final_documents
