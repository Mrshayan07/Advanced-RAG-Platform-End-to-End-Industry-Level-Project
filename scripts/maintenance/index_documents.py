from pathlib import Path

from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.processors.metadata import enrich_metadata
from app.ingestion.processors.splitter import split_documents
from app.rag.embeddings.service import EmbeddingService
from app.rag.vectorstore.qdrant import QdrantService


BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = BASE_DIR / "storage" / "documents"


def load_documents():
    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        loaded_documents = load_pdf_file(str(pdf_path))
        documents.extend(loaded_documents)

    if not documents:
        raise RuntimeError(
            "No PDF documents found in storage/documents."
        )

    chunks = split_documents(documents)
    chunks = enrich_metadata(chunks)

    return chunks


def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Chunks created: {len(documents)}")

    print("\nInitializing embedding service...")
    embedding_service = EmbeddingService()

    print("\nInitializing Qdrant...")
    qdrant_service = QdrantService()

    print("\nCreating collection if needed...")
    qdrant_service.create_collection()

    print("\nUploading vectors...")
    qdrant_service.upsert_documents(
        documents=documents,
        embedding_service=embedding_service,
    )

    count = qdrant_service.client.count(
        collection_name=qdrant_service.collection_name,
        exact=True,
    ).count

    print("\nIndexing completed.")
    print(f"Qdrant points: {count}")


if __name__ == "__main__":
    main()