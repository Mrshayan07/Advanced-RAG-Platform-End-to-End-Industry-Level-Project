from langchain_core.runnables import RunnableLambda


def test_langsmith_trace():
    def process_question(question: str) -> str:
        return f"Processed question: {question}"

    chain = RunnableLambda(process_question)

    result = chain.invoke(
        "What technologies does Shayan know?"
    )

    print("\nResult:")
    print(result)

    assert result == (
        "Processed question: "
        "What technologies does Shayan know?"
    )