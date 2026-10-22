# app/evaluation/eval_context_relevance.py

"""
Context Relevance Evaluation

Metric:
Does the retrieved context actually contain information
relevant to answering the question?

Important:
- Contextual Relevancy is evaluated only on answerable goldens.
- Unanswerable goldens are evaluated separately later using
  retrieval confidence / abstention metrics.
"""

from __future__ import annotations

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from app.evaluation.config import (
    get_eval_settings,
    get_eval_thresholds,
)
from app.evaluation.harness import (
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from app.evaluation.models.groq import GroqJudge
from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.processors.metadata import enrich_metadata
from app.ingestion.processors.splitter import split_documents
from app.rag.embeddings.service import EmbeddingService
from app.rag.graph.nodes import RAGNodes
from app.rag.graph.workflow import (
    build_rag_graph as create_rag_graph,
)
from app.rag.query.rewriting import QueryRewriter
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.bm25 import BM25Service
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.vectorstore.qdrant import QdrantService

load_dotenv()

settings = get_eval_settings()
thresholds = get_eval_thresholds()

GOLDEN_PATH = (
    settings.golden_dataset_dir
    / "retriever"
    / "retrieval_goldens.json"
)

JUDGE_MODEL = GroqJudge()

# Evaluation source of truth:
# Old CV only.
EVALUATION_CV = (
    settings.documents_dir
    / "bc2363a9-2d1d-467d-be92-ab96ab050a26.pdf"
)


def load_documents():
    """Load the selected evaluation document."""

    documents_dir = settings.documents_dir

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    if not EVALUATION_CV.exists():
        raise FileNotFoundError(
            f"Evaluation CV not found: {EVALUATION_CV}"
        )

    documents = load_pdf_file(
        str(EVALUATION_CV)
    )

    if not documents:
        raise RuntimeError(
            f"No content extracted from {EVALUATION_CV}"
        )

    return documents


def build_rag_graph():
    """Build the production RAG graph."""

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()

    documents = load_documents()

    chunks = split_documents(documents)
    chunks = enrich_metadata(chunks)

    bm25_service = BM25Service()
    bm25_service.build_index(chunks)

    reranker = CrossEncoderReranker()
    query_rewriter = QueryRewriter()

    retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    nodes = RAGNodes(
        query_rewriter=query_rewriter,
        retriever=retriever,
    )

    return create_rag_graph(nodes)


def run():
    """Run Contextual Relevancy evaluation."""

    goldens = load_goldens(
        GOLDEN_PATH
    )

    if not goldens:
        raise RuntimeError(
            f"No golden test cases found in {GOLDEN_PATH}"
        )

    print(
        "\nBuilding production RAG graph..."
    )

    graph = build_rag_graph()

    metrics = [
        ContextualRelevancyMetric(
            threshold=thresholds.contextual_relevancy_min,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False,
        ),
    ]

    all_test_cases = []

    answerable_count = 0
    skipped_count = 0

    for index, golden in enumerate(
        goldens,
        start=1,
    ):
        question = golden["question"]
        answerable = golden.get(
            "answerable",
            True,
        )

        print(
            f"\n[{index}/{len(goldens)}] {question}"
        )

        # --------------------------------------------------
        # Unanswerable cases are NOT evaluated with
        # Contextual Relevancy.
        #
        # These cases will later be evaluated using
        # retrieval confidence / abstention metrics.
        # --------------------------------------------------
        if not answerable:
            skipped_count += 1

            print(
                "\nSKIPPED Contextual Relevancy"
            )
            print(
                f"Reason: answerable=false "
                f"({golden['id']})"
            )

            continue

        answerable_count += 1

        result = graph.invoke(
            {
                "question": question,
                "chat_history": "",
            }
        )

        documents = result.get(
            "documents",
            [],
        )

        retrieval_context = [
            document.page_content
            for document in documents
        ]

        print(
            "\n" + "-" * 70
        )
        print(
            "RETRIEVED DOCUMENTS"
        )
        print(
            "-" * 70
        )

        for doc_index, document in enumerate(
            documents,
            start=1,
        ):
            print(
                f"\n--- Document {doc_index} ---"
            )

            print("Metadata:")
            print(
                document.metadata
            )

            print("\nContent:")
            print(
                document.page_content[:1500]
            )

        print(
            "-" * 70
        )

        test_case = LLMTestCase(
            input=question,
            retrieval_context=retrieval_context,
            name=golden["id"],
        )

        all_test_cases.append(
            test_case
        )

    if not all_test_cases:
        raise RuntimeError(
            "No answerable golden cases available "
            "for Contextual Relevancy evaluation."
        )

    print(
        "\n" + "=" * 70
    )
    print(
        "CONTEXT RELEVANCE EVALUATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Answerable cases evaluated: "
        f"{answerable_count}"
    )

    print(
        f"Unanswerable cases skipped: "
        f"{skipped_count}"
    )

    print(
        "=" * 70
    )

    evaluation_result = evaluate(
        test_cases=all_test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            throttle_value=0,
            max_concurrent=1,
        ),
        hyperparameters={
            "retriever": "hybrid_bm25_qdrant_reranked",
            "top_k": settings.retrieval_k,
            "judge_model": JUDGE_MODEL.get_model_name(),
            "golden_set": str(GOLDEN_PATH),
            "evaluation_document": EVALUATION_CV.name,
            "answerable_cases_only": True,
        },
    )

    summary = summarize_by_metric(
        evaluation_result
    )

    return summary


def run_local():
    """Standalone execution."""

    summary = run()

    print_summary(
        "context_relevance",
        summary,
    )

    return summary


if __name__ == "__main__":
    run_local()