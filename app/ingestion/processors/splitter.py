from __future__ import annotations

from langchain_core.documents import Document

SECTION_HEADINGS = {
    "professional summary",
    "experience",
    "projects",
    "technical skills",
    "education",
    "certifications",
    "languages",
    "achievements",
}

SUBSECTION_PREFIXES = {
    "frontend:",
    "backend:",
    "database:",
    "tools:",
    "tech stack:",
}


MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2000
MAX_CHUNK_OVERLAP = 500


def _normalize(line: str) -> str:
    """Normalize a line for heading detection."""
    return " ".join(line.strip().lower().split())


def _is_section_heading(line: str) -> bool:
    """Check whether a line is a major document section."""
    return _normalize(line) in SECTION_HEADINGS


def _is_subsection_heading(line: str) -> bool:
    """Check whether a line is a technical subsection."""
    normalized = _normalize(line)

    return any(
        normalized.startswith(prefix)
        for prefix in SUBSECTION_PREFIXES
    )


def _validate_chunk_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Validate dynamic chunking parameters.

    Rules:

        MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE

        0 <= chunk_overlap <= MAX_CHUNK_OVERLAP

        chunk_overlap < chunk_size
    """

    if not isinstance(chunk_size, int):
        raise ValueError(
            "chunk_size must be an integer."
        )

    if not isinstance(chunk_overlap, int):
        raise ValueError(
            "chunk_overlap must be an integer."
        )

    if not (
        MIN_CHUNK_SIZE
        <= chunk_size
        <= MAX_CHUNK_SIZE
    ):
        raise ValueError(
            f"chunk_size must be between "
            f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}."
        )

    if not (
        0
        <= chunk_overlap
        <= MAX_CHUNK_OVERLAP
    ):
        raise ValueError(
            f"chunk_overlap must be between "
            f"0 and {MAX_CHUNK_OVERLAP}."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


def _split_large_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split oversized text while preserving overlap.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(
            end - chunk_overlap,
            start + 1,
        )

    return chunks


def _build_sections(
    text: str,
) -> list[tuple[str, str]]:
    """
    Build logical sections and technical subsections.

    Example:

        Technical Skills
        Frontend: ...
        Backend: ...
        Database: ...

    becomes:

        Technical Skills / Frontend
        Technical Skills / Backend
        Technical Skills / Database
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sections: list[tuple[str, str]] = []

    current_major = "document"
    current_subsection: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines

        if not current_lines:
            return

        if current_subsection:
            section_name = (
                f"{current_major} / "
                f"{current_subsection}"
            )
        else:
            section_name = current_major

        sections.append(
            (
                section_name,
                "\n".join(current_lines),
            )
        )

        current_lines = []

    for line in lines:

        # -------------------------------------------------
        # Major section
        # -------------------------------------------------

        if _is_section_heading(line):
            flush()

            current_major = line
            current_subsection = None

            continue

        # -------------------------------------------------
        # Technical subsection
        # -------------------------------------------------

        if _is_subsection_heading(line):
            flush()

            normalized = _normalize(line)

            current_subsection = normalized.split(
                ":",
                1,
            )[0]

            # Keep the actual heading line as content.
            current_lines.append(line)

            continue

        # -------------------------------------------------
        # Normal content
        # -------------------------------------------------

        current_lines.append(line)

    flush()

    return sections


def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Production-oriented structure-aware splitter.

    Dynamic parameters:

        chunk_size:
            Maximum character size for oversized chunks.

        chunk_overlap:
            Number of overlapping characters between
            consecutive chunks.

    Strategy:

        1. Detect major document sections.
        2. Detect technical subsections.
        3. Keep small logical sections together.
        4. Split only oversized sections.
        5. Store section information in metadata.
        6. Store chunking configuration in metadata.
    """

    _validate_chunk_parameters(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    final_chunks: list[Document] = []

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        sections = _build_sections(text)

        for section_name, section_text in sections:

            chunks = _split_large_text(
                text=section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk in chunks:

                metadata = dict(
                    document.metadata
                )

                metadata.update(
                    {
                        "section": section_name,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    }
                )

                final_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata,
                    )
                )

    return final_chunks