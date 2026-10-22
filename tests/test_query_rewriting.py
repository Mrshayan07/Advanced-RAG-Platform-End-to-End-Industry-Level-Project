from types import SimpleNamespace

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.rag.query.rewriting import QueryRewriter


def build_test_rewriter(response_text: str) -> QueryRewriter:
    rewriter = QueryRewriter.__new__(QueryRewriter)

    rewriter.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
Conversation history:
{chat_history}

Original user question:
{question}

Rewritten search query:
""",
            )
        ]
    )

    rewriter.llm = RunnableLambda(
        lambda _: SimpleNamespace(
            content=response_text
        )
    )

    return rewriter


def test_rewrite_returns_llm_rewritten_query():
    rewriter = build_test_rewriter(
        "Shayan frontend development technologies frontend skills"
    )

    result = rewriter.rewrite(
        question=(
            "What technologies did Shayan use "
            "for frontend development?"
        )
    )

    assert result == (
        "Shayan frontend development technologies frontend skills"
    )


def test_rewrite_strips_whitespace():
    rewriter = build_test_rewriter(
        "   Shayan Java experience Java programming language skills   "
    )

    result = rewriter.rewrite(
        question="Does Shayan have experience with Java?"
    )

    assert result == (
        "Shayan Java experience Java programming language skills"
    )


def test_rewrite_falls_back_to_original_question_when_response_is_empty():
    rewriter = build_test_rewriter("")

    question = (
        "What is Shayan's primary professional role?"
    )

    result = rewriter.rewrite(
        question=question
    )

    assert result == question


def test_rewrite_converts_non_string_response_to_string():
    rewriter = QueryRewriter.__new__(QueryRewriter)

    rewriter.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "{question}",
            )
        ]
    )

    rewriter.llm = RunnableLambda(
        lambda _: SimpleNamespace(
            content=12345
        )
    )

    result = rewriter.rewrite(
        question="What is Shayan's role?"
    )

    assert result == "12345"


def test_rewrite_with_fallback_returns_rewritten_query_on_success():
    rewriter = build_test_rewriter(
        "Shayan backend development experience"
    )

    result = rewriter.rewrite_with_fallback(
        question="What is Shayan's backend experience?"
    )

    assert result == (
        "Shayan backend development experience"
    )


def test_rewrite_with_fallback_returns_original_question_on_failure():
    rewriter = QueryRewriter.__new__(QueryRewriter)

    def failing_rewrite(
        question: str,
        chat_history: str = "",
    ) -> str:
        raise RuntimeError("Groq unavailable")

    rewriter.rewrite = failing_rewrite

    question = "What technologies does Shayan know?"

    result = rewriter.rewrite_with_fallback(
        question=question
    )

    assert result == question


def test_rewrite_passes_chat_history_to_chain():
    captured = {}

    def fake_llm(prompt_value):
        captured["messages"] = prompt_value.to_messages()

        return SimpleNamespace(
            content="Shayan current frontend experience"
        )

    rewriter = QueryRewriter.__new__(QueryRewriter)

    rewriter.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
Conversation history:
{chat_history}

Original user question:
{question}

Rewritten search query:
""",
            )
        ]
    )

    rewriter.llm = RunnableLambda(fake_llm)

    result = rewriter.rewrite(
        question="What does he do?",
        chat_history=(
            "User previously asked about "
            "Shayan's career."
        ),
    )

    assert result == (
        "Shayan current frontend experience"
    )

    rendered_prompt = captured["messages"][0].content

    assert (
        "What does he do?"
        in rendered_prompt
    )

    assert (
        "User previously asked about Shayan's career."
        in rendered_prompt
    )