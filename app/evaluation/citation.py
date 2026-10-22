from __future__ import annotations

from dataclasses import dataclass

from app.rag.citations.models import Citation, CitationClaim


@dataclass
class CitationEvaluationInput:
    claims: list[CitationClaim]
    citations: list[Citation]


class CitationEvaluator:
    """Evaluates citation correctness and citation completeness."""

    def __init__(
        self,
        claims: list[CitationClaim],
        citations: list[Citation],
    ):
        self.claims = claims
        self.citations = citations

        self.citation_map = {
            citation.citation_id: citation
            for citation in citations
        }

    def calculate_completeness(self) -> float:
        """
        Calculate the percentage of claims
        that contain at least one citation.
        """

        if not self.claims:
            return 1.0

        cited_claims = sum(
            1
            for claim in self.claims
            if claim.citation_ids
        )

        return cited_claims / len(self.claims)

    def get_uncited_claims(self) -> list[CitationClaim]:
        """Return claims without citations."""

        return [
            claim
            for claim in self.claims
            if not claim.citation_ids
        ]

    def get_invalid_citation_ids(self) -> list[int]:
        """Return citation IDs that do not exist."""

        invalid_ids = []

        for claim in self.claims:
            for citation_id in claim.citation_ids:
                if citation_id not in self.citation_map:
                    invalid_ids.append(citation_id)

        return sorted(set(invalid_ids))

    def get_citation(
        self,
        citation_id: int,
    ) -> Citation | None:
        """Return citation by citation ID."""

        return self.citation_map.get(citation_id)