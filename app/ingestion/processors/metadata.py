from pathlib import Path

from langchain_core.documents import Document


def enrich_metadata(
    documents: list[Document],
    original_file_name: str | None = None,
) -> list[Document]:
    total_chunks = len(documents)

    for index, document in enumerate(documents):
        source = document.metadata.get("source", "")

        file_name = (
            original_file_name
            if original_file_name
            else Path(source).name
        )

        document.metadata.update(
            {
                "file_name": file_name,
                "file_type": Path(source).suffix.lstrip(".").lower(),
                "chunk_index": index,
                "total_chunks": total_chunks,
            }
        )

    return documents