from app.evaluation.models.ollama import OllamaJudge
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase


judge = OllamaJudge()

metric = FaithfulnessMetric(
    model=judge,
    threshold=0.7,
    include_reason=True,
    async_mode=False,
)

test_case = LLMTestCase(
    input="What is Shayan Javed's primary professional role?",
    actual_output=(
        "Shayan Javed is a Full Stack Developer, "
        "specializing in MERN-stack development."
    ),
    retrieval_context=[
        "Shayan Javed\nFull Stack Developer | MERN Stack Developer"
    ],
)

print("MODEL:")
print(type(metric.model))
print(metric.model.get_model_name())

print("\nMEASURING...")

try:
    score = metric.measure(test_case)

    print("\nSCORE:")
    print(score)

    print("\nREASON:")
    print(metric.reason)

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(e)