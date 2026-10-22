from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.groq import GroqJudge


judge = GroqJudge()

metric = FaithfulnessMetric(
    threshold=0.70,
    model=judge,
    include_reason=True,
    async_mode=False,
)

test_case = LLMTestCase(
    input="What is Shayan Javed's primary professional role?",
    actual_output=(
        "Shayan Javed's primary professional role is a "
        "Full Stack Developer, specializing in MERN-stack development."
    ),
    retrieval_context=[
        "Shayan Javed\n"
        "Full Stack Developer | MERN Stack Developer"
    ],
)

print("Running Groq + DeepEval Faithfulness...")

metric.measure(test_case)

print("\nSUCCESS")
print("Score:", metric.score)
print("Reason:", metric.reason)