from pydantic import BaseModel


class Citation(BaseModel):
    citation_id: int
    file_name: str
    page: str | None = None
    chunk_index: int | None = None
    source: str


class CitationClaim(BaseModel):
    text: str
    citation_ids: list[int]


class CitationResponse(BaseModel):
    claims: list[CitationClaim]