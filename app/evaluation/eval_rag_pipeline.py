from __future__ import annotations

import math
from pathlib import Path
from uuid import UUID

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.documents import Document

from app.evaluation.config import (
    get_eval_settings,
    get_eval_thresholds,
)
from app.evaluation.harness import (
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from app.evaluation.models.ollama import OllamaJudge
from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.loaders.text import load_text_file
from app.rag.embeddings.service import EmbeddingService
from app.rag.graph.nodes import RAGNodes
from app.rag.graph.workflow import build_rag_graph
from app.rag.query.rewriting import QueryRewriter
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.bm25 import BM25Service
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.vectorstore.qdrant import QdrantService

# ---------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------

load_dotenv()

settings = get_eval_settings()
thresholds = get_eval_thresholds()


# ---------------------------------------------------------------------
# EVALUATION CONFIGURATION
# ---------------------------------------------------------------------

GOLDEN_PATH = (
    settings.golden_dataset_dir
    / "retriever"
    / "retrieval_goldens.json"
)

# Dedicated tenant used ONLY for offline evaluation.
#
# This tenant must never be used by normal application users.
#
# Using a fixed UUID makes the evaluation reproducible across runs.
EVAL_TENANT_ID = UUID(
    "00000000-0000-0000-0000-000000000001"
)


# ---------------------------------------------------------------------
# DEEPEVAL JUDGE
# ---------------------------------------------------------------------

# Qwen3 runs locally through Ollama.
#
# Important:
#
# Gemini = RAG answer generation
# Qwen3 = DeepEval judge
#
# They are intentionally separate.
JUDGE_MODEL = OllamaJudge()


# ---------------------------------------------------------------------
# DOCUMENT LOADING
# ---------------------------------------------------------------------

def load_documents() -> list[Document]:
    """
    Load the local evaluation corpus.

    These files are the offline evaluation dataset.

    The original files do not contain tenant_id metadata.
    Tenant metadata is added later by prepare_evaluation_documents().
    """

    documents: list[Document] = []

    documents_dir = Path(
        settings.documents_dir
    )

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: {documents_dir}"
        )

    for file_path in documents_dir.iterdir():

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            documents.extend(
                load_pdf_file(
                    str(file_path)
                )
            )

        elif suffix in {".txt", ".text"}:
            documents.extend(
                load_text_file(
                    str(file_path)
                )
            )

    if not documents:
        raise ValueError(
            f"No supported documents found in {documents_dir}"
        )

    return documents


# ---------------------------------------------------------------------
# EVALUATION TENANT PREPARATION
# ---------------------------------------------------------------------

def prepare_evaluation_documents(
    documents: list[Document],
    tenant_id: UUID | str,
) -> list[Document]:
    """
    Add the dedicated evaluation tenant_id to every document.

    We create new Document objects instead of mutating the loader
    output directly.

    This keeps the evaluation corpus isolated from the original
    document metadata.
    """

    normalized_tenant_id = str(
        tenant_id
    )

    if not normalized_tenant_id:
        raise ValueError(
            "Evaluation tenant_id cannot be empty."
        )

    prepared_documents: list[Document] = []

    for document in documents:

        metadata = dict(
            document.metadata
        )

        metadata["tenant_id"] = (
            normalized_tenant_id
        )

        metadata["evaluation"] = True

        prepared_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    if not prepared_documents:
        raise ValueError(
            "No evaluation documents were prepared."
        )

    return prepared_documents


# ---------------------------------------------------------------------
# QDRANT EVALUATION CORPUS
# ---------------------------------------------------------------------

def index_evaluation_corpus(
    documents: list[Document],
    tenant_id: UUID | str,
) -> None:
    """
    Index the evaluation corpus into Qdrant under the dedicated
    evaluation tenant.

    Existing production tenants are not modified.

    Because Qdrant point IDs include tenant_id, the evaluation
    corpus gets a separate tenant-scoped set of points.
    """

    if not documents:
        raise ValueError(
            "Cannot index an empty evaluation corpus."
        )

    qdrant_service = QdrantService()
    embedding_service = EmbeddingService()

    qdrant_service.create_collection()

    qdrant_service.upsert_documents(
        documents=documents,
        embedding_service=embedding_service,
        tenant_id=tenant_id,
    )

    print(
        "\n[Evaluation] Qdrant corpus indexed:"
    )

    print(
        f"  tenant_id={tenant_id}"
    )

    print(
        f"  documents={len(documents)}"
    )


# ---------------------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------------------

def build_retriever(
    documents: list[Document],
    tenant_id: UUID | str,
):
    """
    Build the tenant-aware hybrid retriever.

    Retrieval architecture:

        Query
          |
          +--> Dense / Qdrant
          |
          +--> BM25
          |
          +--> RRF
          |
          +--> Cross Encoder
    """

    normalized_tenant_id = str(
        tenant_id
    )

    # ---------------------------------------------------------------
    # BM25
    # ---------------------------------------------------------------

    bm25_service = BM25Service()

    bm25_service.build_index(
        documents=documents,
        tenant_id=normalized_tenant_id,
    )

    # ---------------------------------------------------------------
    # Dense Embeddings
    # ---------------------------------------------------------------

    embedding_service = EmbeddingService()

    # ---------------------------------------------------------------
    # Qdrant
    # ---------------------------------------------------------------

    qdrant_service = QdrantService()

    # ---------------------------------------------------------------
    # Cross Encoder
    # ---------------------------------------------------------------

    reranker = CrossEncoderReranker()

    # ---------------------------------------------------------------
    # Query Rewriter
    # ---------------------------------------------------------------

    query_rewriter = QueryRewriter()

    # ---------------------------------------------------------------
    # Hybrid Retriever
    # ---------------------------------------------------------------

    retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    return (
        retriever,
        query_rewriter,
    )


# ---------------------------------------------------------------------
# RAG GRAPH
# ---------------------------------------------------------------------

def build_rag_pipeline(
    retriever,
    query_rewriter,
):
    """
    Build the LangGraph RAG pipeline.

    RAGNodes obtains tenant_id from graph state during retrieval.
    """

    nodes = RAGNodes(
        query_rewriter=query_rewriter,
        retriever=retriever,
    )

    rag_graph = build_rag_graph(
        nodes
    )

    return rag_graph


# ---------------------------------------------------------------------
# DEEPEVAL METRICS
# ---------------------------------------------------------------------

def build_metrics():
    """
    Build the DeepEval metrics.

    All metrics use local Qwen3:4B through Ollama.

    No Gemini/Groq API quota is used for judging.
    """

    return [
        ContextualRelevancyMetric(
            threshold=(
                thresholds.contextual_relevancy_min
            ),
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        FaithfulnessMetric(
            threshold=(
                thresholds.faithfulness_min
            ),
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        AnswerRelevancyMetric(
            threshold=(
                thresholds.answer_relevancy_min
            ),
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]


# ---------------------------------------------------------------------
# SUMMARY MERGING
# ---------------------------------------------------------------------

def merge_summaries(
    all_summaries,
):
    """
    Merge individual DeepEval summaries into one final summary.
    """

    buckets = {}

    for summary in all_summaries:

        for metric_name, metric_data in summary.items():

            bucket = buckets.setdefault(
                metric_name,
                {
                    "n": 0,
                    "passed": 0,
                    "scores": [],
                },
            )

            n = metric_data["n"]

            bucket["n"] += n

            passed = round(
                (
                    metric_data["pass_rate"]
                    / 100.0
                )
                * n
            )

            bucket["passed"] += passed

            avg_score = (
                metric_data["avg_score"]
            )

            if not math.isnan(avg_score):
                bucket["scores"].extend(
                    [avg_score] * n
                )

    final_summary = {}

    for metric_name, bucket in buckets.items():

        scores = bucket["scores"]

        total = bucket["n"]

        final_summary[metric_name] = {
            "n": total,

            "pass_rate": (
                100
                * bucket["passed"]
                / total
                if total
                else 0.0
            ),

            "avg_score": (
                sum(scores) / len(scores)
                if scores
                else float("nan")
            ),

            "min_score": (
                min(scores)
                if scores
                else float("nan")
            ),

            "max_score": (
                max(scores)
                if scores
                else float("nan")
            ),
        }

    return final_summary


# ---------------------------------------------------------------------
# EVALUATION RUN
# ---------------------------------------------------------------------

def run(
    rag_graph,
    tenant_id: UUID | str,
):
    """
    Run the complete DeepEval evaluation.

    Every RAG graph invocation receives the dedicated evaluation
    tenant_id.

    This is required because the production retrieval layer
    enforces tenant isolation.
    """

    normalized_tenant_id = str(
        tenant_id
    )

    goldens = load_goldens(
        GOLDEN_PATH
    )

    if not goldens:
        raise ValueError(
            f"No goldens found in {GOLDEN_PATH}"
        )

    print(
        f"\nLoaded {len(goldens)} golden questions."
    )

    print(
        f"Evaluation tenant: {normalized_tenant_id}"
    )

    all_summaries = []

    metrics = build_metrics()

    for index, golden in enumerate(
        goldens,
        start=1,
    ):

        query = golden["question"]

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"[{index}/{len(goldens)}] Evaluating:"
        )

        print(query)

        print(
            "=" * 70
        )

        # -------------------------------------------------------------
        # RAG EXECUTION
        # -------------------------------------------------------------

        result = rag_graph.invoke(
            {
                "question": query,

                "chat_history": "",

                # Critical tenant context.
                "tenant_id": normalized_tenant_id,
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

        actual_output = result.get(
            "answer"
        )

        if not actual_output:
            raise ValueError(
                "RAG pipeline did not return an answer.\n"
                f"Question: {query}\n"
                f"Available state keys: "
                f"{list(result.keys())}"
            )

        print(
            f"Retrieved documents: "
            f"{len(documents)}"
        )

        print(
            "Generated answer: "
            f"{actual_output[:200]}"
            f"{'...' if len(actual_output) > 200 else ''}"
        )

        # -------------------------------------------------------------
        # DEEPEVAL TEST CASE
        # -------------------------------------------------------------

        test_case = LLMTestCase(
            input=query,

            actual_output=actual_output,

            retrieval_context=retrieval_context,

            expected_output=golden.get(
                "ideal_answer"
            ),
        )

        print(
            "\nRunning DeepEval for this test case..."
        )

        # -------------------------------------------------------------
        # DEEPEVAL
        # -------------------------------------------------------------

        evaluation_result = evaluate(
            test_cases=[
                test_case
            ],

            async_config=AsyncConfig(
                run_async=False,
            ),

            metrics=metrics,

            hyperparameters={
                "pipeline": (
                    "query_rewriting"
                    "_hybrid_bm25_qdrant"
                    "_rrf"
                    "_cross_encoder"
                    "_gemini_generation"
                ),

                "top_k": 3,

                "judge_model": (
                    JUDGE_MODEL.get_model_name()
                ),

                "golden_set": str(
                    GOLDEN_PATH
                ),

                "test_case": golden["id"],

                "evaluation_tenant": (
                    normalized_tenant_id
                ),
            },
        )

        single_summary = (
            summarize_by_metric(
                evaluation_result
            )
        )

        all_summaries.append(
            single_summary
        )

        print_summary(
            f"Test case: {golden['id']}",
            single_summary,
        )

    final_summary = merge_summaries(
        all_summaries
    )

    return final_summary


# ---------------------------------------------------------------------
# LOCAL EVALUATION ENTRY POINT
# ---------------------------------------------------------------------

def run_local():
    """
    Prepare and execute the complete local RAG evaluation.

    Flow:

        Local documents
              ↓
        Evaluation tenant metadata
              ↓
        Qdrant indexing
              ↓
        BM25 indexing
              ↓
        Hybrid retriever
              ↓
        LangGraph RAG
              ↓
        Gemini generation
              ↓
        DeepEval
              ↓
        Qwen3:4B judge
    """

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------

    raw_documents = load_documents()

    print(
        f"\nLoaded {len(raw_documents)} "
        "documents/chunks for RAG evaluation."
    )

    # ---------------------------------------------------------------
    # PREPARE TENANT METADATA
    # ---------------------------------------------------------------

    evaluation_documents = (
        prepare_evaluation_documents(
            documents=raw_documents,
            tenant_id=EVAL_TENANT_ID,
        )
    )

    print(
        "\nPrepared evaluation corpus:"
    )

    print(
        f"  tenant_id={EVAL_TENANT_ID}"
    )

    print(
        f"  documents={len(evaluation_documents)}"
    )

    # ---------------------------------------------------------------
    # INDEX INTO QDRANT
    # ---------------------------------------------------------------

    index_evaluation_corpus(
        documents=evaluation_documents,
        tenant_id=EVAL_TENANT_ID,
    )

    # ---------------------------------------------------------------
    # BUILD RETRIEVER
    # ---------------------------------------------------------------

    retriever, query_rewriter = (
        build_retriever(
            documents=evaluation_documents,
            tenant_id=EVAL_TENANT_ID,
        )
    )

    # ---------------------------------------------------------------
    # BUILD LANGGRAPH
    # ---------------------------------------------------------------

    rag_graph = build_rag_pipeline(
        retriever=retriever,
        query_rewriter=query_rewriter,
    )

    # ---------------------------------------------------------------
    # RUN EVALUATION
    # ---------------------------------------------------------------

    return run(
        rag_graph=rag_graph,
        tenant_id=EVAL_TENANT_ID,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    summary = run_local()

    print_summary(
        "rag_triad",
        summary,
    )
