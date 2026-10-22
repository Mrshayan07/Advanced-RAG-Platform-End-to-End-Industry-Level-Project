from langsmith import Client, traceable


client = Client()


@traceable(
    name="langsmith_connection_test",
    run_type="chain",
)
def test_trace(question: str) -> str:
    return f"Processed question: {question}"


if __name__ == "__main__":
    print("=" * 70)
    print("LANGSMITH CONNECTION TEST")
    print("=" * 70)

    print(
        f"LangSmith project: "
        f"{client.tracing_sample_rate}"
    )

    result = test_trace(
        "What databases were used?"
    )

    print(
        f"Result: {result}"
    )

    print("=" * 70)
    print("TRACE TEST COMPLETED")
    print("=" * 70)