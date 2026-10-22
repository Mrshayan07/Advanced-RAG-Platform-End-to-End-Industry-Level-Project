from unittest.mock import patch

from app.api.rag import run_rag_graph


def test_langsmith_trace():
    expected_result = {
        "answer": "Shayan knows Python, SQL, Machine Learning, Deep Learning, NLP, Transformers, and RAG.",
        "citations": [],
    }

    class FakeGraph:
        def invoke(self, state, config):
            assert state["question"] == (
                "What technologies does Shayan know?"
            )

            assert state["chat_history"] == (
                "Previous conversation about AI."
            )

            assert state["user_memory"] == [
                {
                    "key": "career_goal",
                    "value": "AI Engineer",
                    "memory_type": "preference",
                }
            ]

            assert state["tenant_id"] == "tenant-123"

            assert "tags" in config
            assert "rag-api" in config["tags"]
            assert "production-rag" in config["tags"]

            assert config["metadata"]["user_id"] == "user-123"
            assert config["metadata"]["tenant_id"] == "tenant-123"
            assert config["metadata"]["role"] == "user"

            return expected_result

    with patch(
        "app.api.rag.get_rag_graph",
        return_value=FakeGraph(),
    ):
        result = run_rag_graph(
            question="What technologies does Shayan know?",
            chat_history="Previous conversation about AI.",
            user_memory=[
                {
                    "key": "career_goal",
                    "value": "AI Engineer",
                    "memory_type": "preference",
                }
            ],
            tenant_id="tenant-123",
            user_id="user-123",
            role="user",
        )

    assert result == expected_result