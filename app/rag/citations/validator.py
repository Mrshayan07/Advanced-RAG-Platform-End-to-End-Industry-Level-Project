from langchain_core.documents import Document

from app.rag.citations.models import CitationClaim


class CitationValidator:
    """Validates citation references attached to generated claims."""

    def validate(
        self,
        claims: list[CitationClaim],
        documents: list[Document],
    ) -> list[CitationClaim]:

        valid_claims = []

        for claim in claims:
            valid_citation_ids = []

            for citation_id in claim.citation_ids:
                if 1 <= citation_id <= len(documents):
                    valid_citation_ids.append(citation_id)

            valid_claims.append(
                claim.model_copy(
                    update={
                        "citation_ids": valid_citation_ids,
                    }
                )
            )

        return valid_claims