# app/evaluation/eval_robustness.py
"""
Robustness Evaluation - Hallucination Detection

Metric: Does the system correctly handle unanswerable questions?

For unanswerable questions: system MUST refuse or say info not found
For answerable questions: system MUST NOT wrongly refuse

This is a production-critical safety check.
"""

from __future__ import annotations

from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
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
from app.ingestion.loaders.text import load_text_file
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


def load_documents():
    """Load documents from storage/documents."""

    documents_dir = settings.documents_dir

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    documents = []

    for file_path in documents_dir.iterdir():

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(
                load_pdf_file(str(file_path))
            )

        elif suffix in {".txt", ".text"}:
            documents.extend(
                load_text_file(str(file_path))
            )

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {documents_dir}"
        )

    return documents


def build_rag_graph():
    """Build the production RAG graph."""

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    documents = load_documents()

    bm25_service = BM25Service()
    bm25_service.build_index(documents)

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
    """Run robustness (hallucination) evaluation."""

    goldens = load_goldens(GOLDEN_PATH)

    if not goldens:
        raise RuntimeError(
            f"No golden test cases found in {GOLDEN_PATH}"
        )

    print("\nBuilding production RAG graph...")

    graph = build_rag_graph()

    # DeepEval's HallucinationMetric: checks if answer contains
    # facts NOT grounded in the retrieval context
    metrics = [
        HallucinationMetric(
            threshold=thresholds.faithfulness_min,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False,
        ),
    ]

    all_test_cases = []

    for index, golden in enumerate(goldens, start=1):

        question = golden["question"]
        answerable = golden.get("answerable", True)

        result = graph.invoke(
            {
                "question": question,
                "chat_history": "",
            }
        )

        answer = result.get("answer", "")
        documents = result.get("documents", [])
        retrieval_context = [
            document.page_content
            for document in documents
        ]

        # For unanswerable questions, ideal answer should indicate
        # that info is not available
        ideal_answer = golden.get("ideal_answer", "")

        print(
            f"\n[{index}/{len(goldens)}] "
            f"{'[UNANSWERABLE]' if not answerable else '[ANSWERABLE]'} "
            f"{question}"
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ideal_answer,
            retrieval_context=retrieval_context,
            name=golden["id"],
        )

        all_test_cases.append(test_case)

    print("\n" + "=" * 70)
    print("Running DeepEval (Robustness - Hallucination Detection)...")
    print("=" * 70)

    evaluation_result = evaluate(
        test_cases=all_test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "hybrid_bm25_qdrant_reranked",
            "top_k": settings.retrieval_k,
            "judge_model": JUDGE_MODEL.get_model_name(),
            "golden_set": str(GOLDEN_PATH),
            "test_type": "robustness_hallucination",
        },
    )

    summary = summarize_by_metric(evaluation_result)

    return summary


def run_local():
    """Standalone execution."""

    summary = run()

    print_summary("robustness", summary)

    return summary


if __name__ == "__main__":
    run_local()