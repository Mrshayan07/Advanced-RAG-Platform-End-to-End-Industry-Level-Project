from app.api.rag import rag_graph
from app.rag.graph.nodes import RAGNodes


TENANT_A = "0ee91af3-637a-4000-9558-855f04cd6325"
TENANT_B = "fb86c917-3130-41fe-b7ff-31ce158e81a9"


def get_retriever():
    """
    Extract the RAGNodes instance from the compiled graph
    so we can test retrieval directly without LLM generation.
    """

    # The graph nodes are compiled inside LangGraph.
    # We use the original pipeline construction instead
    # to get access to the HybridRetriever instance.
    from app.api.rag import build_rag_pipeline
    from app.rag.query.rewriting import QueryRewriter
    from app.rag.retrieval.bm25 import BM25Service
    from app.rag.retrieval.hybrid import HybridRetriever
    from app.rag.embeddings.service import EmbeddingService
    from app.rag.reranking.cross_encoder import CrossEncoderReranker
    from app.rag.vectorstore.qdrant import QdrantService
    from collections import defaultdict

    query_rewriter = QueryRewriter()
    bm25_service = BM25Service()
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()

    documents = qdrant_service.get_all_documents()

    tenant_documents = defaultdict(list)

    for document in documents:
        tenant_id = document.metadata.get("tenant_id")

        if tenant_id:
            tenant_documents[str(tenant_id)].append(document)

    for tenant_id, tenant_docs in tenant_documents.items():
        bm25_service.build_index(
            documents=tenant_docs,
            tenant_id=tenant_id,
        )

    reranker = CrossEncoderReranker()

    return HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )


def test_tenant_retrieval(
    retriever,
    tenant_id: str,
    question: str,
):
    print("\n" + "=" * 80)
    print(f"TESTING TENANT: {tenant_id}")
    print("=" * 80)

    print(f"Question: {question}")

    documents = retriever.search(
        query=question,
        chat_history="",
        dense_k=10,
        bm25_k=10,
        top_k=3,
        tenant_id=tenant_id,
    )

    print(f"\nRetrieved documents: {len(documents)}")

    if not documents:
        print("No documents retrieved.")
        return True

    isolation_passed = True

    for index, document in enumerate(
        documents,
        start=1,
    ):
        document_tenant = str(
            document.metadata.get("tenant_id")
        )

        file_name = document.metadata.get(
            "file_name",
            "Unknown",
        )

        print(f"\nDocument {index}")
        print(f"File: {file_name}")
        print(f"Document tenant: {document_tenant}")

        if document_tenant != tenant_id:
            isolation_passed = False
            print("❌ TENANT ISOLATION FAILED")
        else:
            print("✅ Tenant matches")

    return isolation_passed


def main():
    print("\n" + "#" * 80)
    print("RAG TENANT ISOLATION RUNTIME TEST")
    print("#" * 80)

    retriever = get_retriever()

    question = "What is Shayan Javed's work experience?"

    tenant_a_passed = test_tenant_retrieval(
        retriever=retriever,
        tenant_id=TENANT_A,
        question=question,
    )

    tenant_b_passed = test_tenant_retrieval(
        retriever=retriever,
        tenant_id=TENANT_B,
        question=question,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if tenant_a_passed and tenant_b_passed:
        print("✅ RAG TENANT ISOLATION PASSED")
        print("Every retrieved document belonged to the requested tenant.")
    else:
        print("❌ RAG TENANT ISOLATION FAILED")
        print("At least one retrieved document belonged to another tenant.")

    print("=" * 80)


if __name__ == "__main__":
    main()