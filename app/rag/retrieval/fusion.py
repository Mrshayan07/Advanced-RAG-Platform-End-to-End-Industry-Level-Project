from collections import defaultdict

from langchain_core.documents import Document


class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        result_lists: list[list[Document]],
        top_k: int = 5,
    ) -> list[Document]:
        scores = defaultdict(float)
        documents = {}

        for results in result_lists:
            for rank, document in enumerate(results, start=1):
                document_id = self._get_document_id(document)

                scores[document_id] += 1 / (
                    self.k + rank
                )

                # Keep the first version of the document.
                if document_id not in documents:
                    documents[document_id] = document

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            documents[document_id]
            for document_id in ranked_ids[:top_k]
        ]

    @staticmethod
    def _get_document_id(document: Document) -> str:
        """
        Generate a stable logical ID for a document chunk.

        We intentionally avoid using the full `source` path because
        the same file can appear with different absolute/relative
        paths between BM25 and Qdrant retrieval.

        Example:
            storage\\documents\\cv.pdf
            C:\\Users\\DELL\\Desktop\\advanced-rag\\storage\\documents\\cv.pdf

        Both should represent the same logical chunk.
        """

        file_name = document.metadata.get("file_name", "")
        chunk_index = document.metadata.get("chunk_index", 0)

        return f"{file_name}:{chunk_index}"