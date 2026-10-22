from langchain_core.documents import Document

from app.rag.citations.models import Citation


class CitationService:
    def create_citations(
        self,
        documents: list[Document],
    ) -> list[Citation]:

        citations = []

        for index, document in enumerate(documents, start=1):
            metadata = document.metadata

            citations.append(
                Citation(
                    citation_id=index,
                    file_name=metadata.get(
                        "file_name",
                        "Unknown",
                    ),
                    page=metadata.get("page_label"),
                    chunk_index=metadata.get("chunk_index"),
                    source=metadata.get(
                        "source",
                        "",
                    ),
                )
            )

        return citations

    def format_context(
        self,
        documents: list[Document],
    ) -> str:

        context_parts = []

        for index, document in enumerate(documents, start=1):
            metadata = document.metadata

            file_name = metadata.get(
                "file_name",
                "Unknown",
            )

            page = metadata.get(
                "page_label",
                "Unknown",
            )

            chunk_index = metadata.get(
                "chunk_index",
                "Unknown",
            )

            context_parts.append(
                f"""
[{index}]
Source: {file_name}
Page: {page}
Chunk: {chunk_index}

Content:
{document.page_content}
"""
            )

        return "\n\n".join(context_parts)