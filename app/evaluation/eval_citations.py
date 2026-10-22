from __future__ import annotations

from dotenv import load_dotenv

from app.evaluation.citation import CitationEvaluator
from app.evaluation.citation_correctness import (
    CitationCorrectnessEvaluator,
)
from app.evaluation.config import (
    get_eval_settings,
    get_eval_thresholds,
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


def load_documents():

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

    questions = [
        "What is Shayan Javed's primary professional role?",
        "What technologies did Shayan use for frontend development?",
        "Does Shayan have experience with Java?",
    ]

    graph = build_rag_graph()

    judge = GroqJudge()

    correctness_evaluator = CitationCorrectnessEvaluator(
        judge=judge,
        threshold=thresholds.citation_correctness_min,
    )

    total_completeness = 0.0
    correctness_scores = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"Citation Evaluation "
            f"{index}/{len(questions)}"
        )
        print("=" * 70)

        print(f"Question: {question}")

        result = graph.invoke(
            {
                "question": question,
                "chat_history": "",
            }
        )

        claims = result.get("claims", [])
        citations = result.get("citations", [])

        print("\nGenerated answer:")
        print(result.get("answer", ""))

        print("\nClaims:")

        for claim in claims:

            print(f"- {claim.text}")
            print(
                f"  Citation IDs: "
                f"{claim.citation_ids}"
            )

        citation_evaluator = CitationEvaluator(
            claims=claims,
            citations=citations,
        )

        completeness = (
            citation_evaluator.calculate_completeness()
        )

        total_completeness += completeness

        print(
            f"\nCitation Completeness: "
            f"{completeness:.2f}"
        )

        print("\nCitation Correctness:")

        for claim in claims:

            citation_sources = []

            for citation_id in claim.citation_ids:

                citation = citation_evaluator.get_citation(
                    citation_id
                )

                if citation is None:
                    continue

                citation_sources.append(
                    citation.source
                )

            evaluation = (
                correctness_evaluator.evaluate_claim(
                    claim=claim,
                    citation_sources=citation_sources,
                )
            )

            score = evaluation["score"]

            correctness_scores.append(score)

            print(
                f"- Claim: {evaluation['claim']}"
            )

            print(
                f"  Citation IDs: "
                f"{evaluation['citation_ids']}"
            )

            print(
                f"  Score: {score:.2f}"
            )

            print(
                f"  Status: "
                f"{'PASS' if evaluation['passed'] else 'FAIL'}"
            )

            print(
                f"  Reason: "
                f"{evaluation['reason']}"
            )

    average_completeness = (
        total_completeness / len(questions)
    )

    average_correctness = (
        sum(correctness_scores)
        / len(correctness_scores)
        if correctness_scores
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Citation Evaluation Summary")
    print("=" * 70)

    print(
        f"Average Citation Completeness: "
        f"{average_completeness:.2f}"
    )

    print(
        f"Completeness Threshold: "
        f"{thresholds.citation_completeness_min:.2f}"
    )

    print(
        f"Average Citation Correctness: "
        f"{average_correctness:.2f}"
    )

    print(
        f"Correctness Threshold: "
        f"{thresholds.citation_correctness_min:.2f}"
    )

    completeness_passed = (
        average_completeness
        >= thresholds.citation_completeness_min
    )

    correctness_passed = (
        average_correctness
        >= thresholds.citation_correctness_min
    )

    print(
        f"\nCompleteness: "
        f"{'PASS' if completeness_passed else 'FAIL'}"
    )

    print(
        f"Correctness: "
        f"{'PASS' if correctness_passed else 'FAIL'}"
    )

    print("=" * 70)

    return {
        "completeness": average_completeness,
        "correctness": average_correctness,
    }


if __name__ == "__main__":
    run()