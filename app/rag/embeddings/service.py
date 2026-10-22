from time import perf_counter

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings

settings = get_settings()


class EmbeddingService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    def embed_query(self, text: str) -> list[float]:
        start = perf_counter()

        vector = self.embeddings.embed_query(text)

        elapsed = perf_counter() - start

        print(
            f"[Embedding] Query embedding: "
            f"{elapsed:.3f}s"
        )

        return vector

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        start = perf_counter()

        vectors = self.embeddings.embed_documents(
            texts
        )

        elapsed = perf_counter() - start

        print(
            f"[Embedding] Document embeddings "
            f"({len(texts)} docs): "
            f"{elapsed:.3f}s"
        )

        return vectors