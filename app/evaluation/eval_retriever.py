from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from app.evaluation.config import get_eval_settings, get_eval_thresholds
from app.evaluation.harness import (
    load_goldens,
    print_summary,
    summarize_by_metric,
)
from app.evaluation.models.gemini import GeminiJudge
from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.loaders.text import load_text_file
from app.rag.embeddings.service import EmbeddingService
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

# JUDGE_MODEL = GroqJudge()
JUDGE_MODEL = GeminiJudge()


def load_documents():
    """
    Load supported documents from storage/documents.

    Supported file types:
    - PDF
    - TXT
    """

    documents = []

    documents_dir = Path(settings.documents_dir)

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
                load_pdf_file(str(file_path))
            )

        elif suffix in {".txt", ".text"}:
            documents.extend(
                load_text_file(str(file_path))
            )

    if not documents:
        raise ValueError(
            f"No supported documents found in {documents_dir}"
        )

    return documents


def build_retriever(documents):
    """
    Build the project's hybrid retrieval pipeline.

    BM25:
        Uses documents loaded from storage.

    Dense retrieval:
        Uses Qdrant.

    Fusion:
        Reciprocal Rank Fusion (RRF).

    Reranking:
        Cross-encoder reranker.
    """

    bm25_service = BM25Service()
    bm25_service.build_index(documents)

    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    reranker = CrossEncoderReranker()
    query_rewriter = QueryRewriter()

    retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    return retriever


def run(retriever):
    """
    Run retrieval evaluation against the golden dataset.
    """

    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    for golden in goldens:

        query = golden["question"]

        retrieved_documents = retriever.search(
            query=query,
            chat_history="",
            dense_k=settings.retrieval_k,
            bm25_k=settings.retrieval_k,
            top_k=settings.retrieval_k,
        )

        retrieval_context = [
            document.page_content
            for document in retrieved_documents
        ]

        test_cases.append(
            LLMTestCase(
                input=query,
                expected_output=golden["ideal_answer"],
                actual_output=(
                    "(generator not evaluated in this run)"
                ),
                retrieval_context=retrieval_context,
            )
        )

    metrics = [
        ContextualRecallMetric(
            threshold=thresholds.contextual_recall_min,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=thresholds.contextual_precision_min,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": (
                "hybrid_bm25_qdrant_rrf_reranked"
            ),
            "top_k": settings.retrieval_k,
            "judge_model": (
                f"groq/{JUDGE_MODEL.get_model_name()}"
            ),
            "golden_set": str(GOLDEN_PATH),
        },
    )

    return summarize_by_metric(result)


def run_local():
    """
    Load documents, build the hybrid retriever,
    then evaluate retrieval.
    """

    documents = load_documents()

    print(
        f"\nLoaded {len(documents)} documents/chunks "
        f"for BM25 evaluation."
    )

    retriever = build_retriever(documents)

    return run(retriever)


if __name__ == "__main__":
    summary = run_local()

    print_summary(
        "retriever",
        summary,
    )