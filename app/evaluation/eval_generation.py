from __future__ import annotations

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
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


# ------------------------------------------------------------------
# DeepEval Judge
# ------------------------------------------------------------------

JUDGE_MODEL = GroqJudge()


# ------------------------------------------------------------------
# Load documents
# ------------------------------------------------------------------

def load_documents():
    """
    Load documents from storage/documents.

    These documents are used to build the BM25 index.
    Qdrant remains the source for dense retrieval.
    """

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


# ------------------------------------------------------------------
# Build production RAG graph
# ------------------------------------------------------------------

def build_rag_graph():
    """
    Build the same production RAG graph
    used by the application.
    """

    embedding_service = EmbeddingService()

    qdrant_service = QdrantService()

    documents = load_documents()

    # Build BM25 index
    bm25_service = BM25Service()
    bm25_service.build_index(documents)

    # Cross-encoder reranker
    reranker = CrossEncoderReranker()

    # Query rewriting
    query_rewriter = QueryRewriter()

    # Hybrid retriever
    retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    # RAG nodes
    nodes = RAGNodes(
        query_rewriter=query_rewriter,
        retriever=retriever,
    )

    # LangGraph
    return create_rag_graph(nodes)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def run():

    # --------------------------------------------------------------
    # 1. Load golden dataset
    # --------------------------------------------------------------

    goldens = load_goldens(GOLDEN_PATH)

    if not goldens:
        raise RuntimeError(
            f"No golden test cases found in {GOLDEN_PATH}"
        )

    # --------------------------------------------------------------
    # 2. Build production RAG graph
    # --------------------------------------------------------------

    print("\nBuilding production RAG graph...")

    graph = build_rag_graph()

    # --------------------------------------------------------------
    # 3. Create DeepEval metrics
    # --------------------------------------------------------------

    metrics = [
        FaithfulnessMetric(
            threshold=thresholds.faithfulness_min,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False,
        ),

        AnswerRelevancyMetric(
            threshold=thresholds.answer_relevancy_min,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False,
        ),
    ]

    # --------------------------------------------------------------
    # 4. Create all DeepEval test cases
    # --------------------------------------------------------------

    all_test_cases = []

    for index, golden in enumerate(
        goldens,
        start=1,
    ):

        question = golden["question"]

        # ----------------------------------------------------------
        # Run production RAG
        # ----------------------------------------------------------

        result = graph.invoke(
            {
                "question": question,
                "chat_history": "",
            }
        )

        answer = result.get(
            "answer",
            "",
        )

        documents = result.get(
            "documents",
            [],
        )

        retrieval_context = [
            document.page_content
            for document in documents
        ]

        # ----------------------------------------------------------
        # Display RAG result
        # ----------------------------------------------------------

        print("\n" + "=" * 70)

        print(
            f"Evaluation test case "
            f"{index}/{len(goldens)}"
        )

        print(
            f"ID: {golden['id']}"
        )

        print(
            f"Question: {question}"
        )

        print("\nGenerated answer:")

        print(answer)

        print("\nRetrieved documents:")

        for doc_index, document in enumerate(
            documents,
            start=1,
        ):

            print(
                f"{doc_index}. "
                f"{document.metadata.get('file_name')} "
                f"| chunk="
                f"{document.metadata.get('chunk_index')}"
            )

        # ----------------------------------------------------------
        # Create DeepEval test case
        # ----------------------------------------------------------

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=golden["ideal_answer"],
            retrieval_context=retrieval_context,
            name=golden["id"],
        )

        all_test_cases.append(test_case)

    # --------------------------------------------------------------
    # 5. Run DeepEval ONCE for all test cases
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Running DeepEval...")
    print("=" * 70)

    evaluation_result = evaluate(
        test_cases=all_test_cases,
        metrics=metrics,

        # Run sequentially.
        # This is safer for the Groq judge and avoids
        # unnecessary concurrent requests.
        async_config=AsyncConfig(
            run_async=False,
            throttle_value=0,
            max_concurrent=1,
        ),

        hyperparameters={
            "rag_pipeline": (
                "langgraph_rewrite_hybrid_retrieval_generation"
            ),
            "retrieval_k": settings.retrieval_k,
            "judge_model": (
                JUDGE_MODEL.get_model_name()
            ),
            "golden_set": str(GOLDEN_PATH),
        },
    )

    # --------------------------------------------------------------
    # 6. Summarize evaluation results
    # --------------------------------------------------------------

    summary = summarize_by_metric(
        evaluation_result
    )

    return summary


# ------------------------------------------------------------------
# Local execution
# ------------------------------------------------------------------

def run_local():

    summary = run()

    print_summary(
        "generation",
        summary,
    )

    return summary


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_local()