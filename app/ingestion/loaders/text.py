from pathlib import Path

from langchain_core.documents import Document


def load_text_file(file_path: str) -> list[Document]:
    path = Path(file_path)

    content = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        Document(
            page_content=content,
            metadata={
                "source": str(path),
                "file_type": "text",
            },
        )
    ]