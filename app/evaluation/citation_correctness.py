from __future__ import annotations

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.groq import GroqJudge
from app.rag.citations.models import CitationClaim


class CitationCorrectnessEvaluator:
    """
    Evaluates whether each citation actually supports
    the claim it is attached to.
    """

    def __init__(
        self,
        judge: GroqJudge,
        threshold: float = 0.70,
    ):
        self.judge = judge
        self.threshold = threshold

    def evaluate_claim(
        self,
        claim: CitationClaim,
        citation_sources: list[str],
    ) -> dict:

        if not claim.citation_ids:
            return {
                "claim": claim.text,
                "citation_ids": [],
                "score": 0.0,
                "passed": False,
                "reason": "Claim has no citation.",
            }

        metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.judge,
            include_reason=True,
            async_mode=False,
        )

        test_case = LLMTestCase(
            input=claim.text,
            actual_output=claim.text,
            retrieval_context=citation_sources,
        )

        metric.measure(test_case)

        return {
            "claim": claim.text,
            "citation_ids": claim.citation_ids,
            "score": metric.score,
            "passed": metric.success,
            "reason": metric.reason,
        }