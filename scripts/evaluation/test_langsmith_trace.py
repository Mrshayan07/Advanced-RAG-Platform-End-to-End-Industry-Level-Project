from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnableLambda


def process_question(question: str) -> str:
    return f"Processed question: {question}"


chain = RunnableLambda(process_question)

result = chain.invoke(
    "What is retrieval augmented generation?",
    config={
        "tags": ["langsmith-test"],
        "metadata": {
            "test": "advanced-rag",
        },
    },
)

print(result)