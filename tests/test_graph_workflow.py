from unittest.mock import MagicMock

from app.rag.graph.state import RAGState
from app.rag.graph.workflow import build_rag_graph

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def build_mock_nodes():
    nodes = MagicMock()

    nodes.rewrite.side_effect = lambda state: {
        **state,
        "rewritten_query": "rewritten question",
    }

    nodes.retrieve.side_effect = lambda state: {
        **state,
        "documents": [],
    }

    nodes.generate.side_effect = lambda state: {
        **state,
        "answer": "Generated answer.",
        "claims": [],
        "citations": [],
    }

    return nodes


def initial_state() -> RAGState:
    return {
        "tenant_id": "tenant-a",
        "question": "What technologies does Shayan use?",
        "chat_history": "",
        "user_memory": [],
    }


# ---------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------


def test_build_rag_graph_returns_compiled_graph():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    assert graph is not None


def test_build_rag_graph_contains_expected_nodes():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    node_names = set(graph_data.nodes.keys())

    assert "rewrite" in node_names
    assert "retrieve" in node_names
    assert "generate" in node_names


def test_build_rag_graph_does_not_add_unexpected_processing_nodes():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    node_names = set(graph_data.nodes.keys())

    assert node_names == {
        "__start__",
        "rewrite",
        "retrieve",
        "generate",
        "__end__",
    }


# ---------------------------------------------------------
# Graph Structure
# ---------------------------------------------------------


def test_graph_has_start_to_rewrite_edge():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph_data.edges
    }

    assert (
        "__start__",
        "rewrite",
    ) in edges


def test_graph_has_rewrite_to_retrieve_edge():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph_data.edges
    }

    assert (
        "rewrite",
        "retrieve",
    ) in edges


def test_graph_has_retrieve_to_generate_edge():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph_data.edges
    }

    assert (
        "retrieve",
        "generate",
    ) in edges


def test_graph_has_generate_to_end_edge():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph_data.edges
    }

    assert (
        "generate",
        "__end__",
    ) in edges


def test_graph_has_exactly_four_processing_edges():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph_data = graph.get_graph()

    edges = {
        (edge.source, edge.target)
        for edge in graph_data.edges
    }

    expected_edges = {
        ("__start__", "rewrite"),
        ("rewrite", "retrieve"),
        ("retrieve", "generate"),
        ("generate", "__end__"),
    }

    assert edges == expected_edges


# ---------------------------------------------------------
# End-to-End Graph Invocation
# ---------------------------------------------------------


def test_graph_executes_nodes_in_correct_order():
    nodes = build_mock_nodes()

    execution_order = []

    def rewrite(state):
        execution_order.append("rewrite")

        return {
            **state,
            "rewritten_query": "rewritten question",
        }

    def retrieve(state):
        execution_order.append("retrieve")

        return {
            **state,
            "documents": [],
        }

    def generate(state):
        execution_order.append("generate")

        return {
            **state,
            "answer": "Generated answer.",
            "claims": [],
            "citations": [],
        }

    nodes.rewrite.side_effect = rewrite
    nodes.retrieve.side_effect = retrieve
    nodes.generate.side_effect = generate

    graph = build_rag_graph(nodes)

    result = graph.invoke(
        initial_state()
    )

    assert execution_order == [
        "rewrite",
        "retrieve",
        "generate",
    ]

    assert result["answer"] == (
        "Generated answer."
    )


def test_graph_passes_rewritten_query_to_retrieve():
    nodes = build_mock_nodes()

    def rewrite(state):
        return {
            **state,
            "rewritten_query": "Shayan frontend technologies",
        }

    def retrieve(state):
        assert state["rewritten_query"] == (
            "Shayan frontend technologies"
        )

        return {
            **state,
            "documents": [],
        }

    nodes.rewrite.side_effect = rewrite
    nodes.retrieve.side_effect = retrieve

    graph = build_rag_graph(nodes)

    result = graph.invoke(
        initial_state()
    )

    assert result["rewritten_query"] == (
        "Shayan frontend technologies"
    )


def test_graph_passes_retrieved_documents_to_generate():
    nodes = build_mock_nodes()

    documents = [
        {
            "content": "Document one",
        },
        {
            "content": "Document two",
        },
    ]

    def retrieve(state):
        return {
            **state,
            "documents": documents,
        }

    def generate(state):
        assert state["documents"] == documents

        return {
            **state,
            "answer": "Answer based on documents.",
            "claims": [],
            "citations": [],
        }

    nodes.retrieve.side_effect = retrieve
    nodes.generate.side_effect = generate

    graph = build_rag_graph(nodes)

    result = graph.invoke(
        initial_state()
    )

    assert result["documents"] == documents
    assert result["answer"] == (
        "Answer based on documents."
    )


def test_graph_returns_final_state():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    result = graph.invoke(
        initial_state()
    )

    assert result["tenant_id"] == "tenant-a"
    assert result["question"] == (
        "What technologies does Shayan use?"
    )
    assert result["chat_history"] == ""
    assert result["user_memory"] == []

    assert result["rewritten_query"] == (
        "rewritten question"
    )

    assert result["documents"] == []

    assert result["answer"] == (
        "Generated answer."
    )

    assert result["claims"] == []
    assert result["citations"] == []


def test_graph_calls_each_node_exactly_once():
    nodes = build_mock_nodes()

    graph = build_rag_graph(nodes)

    graph.invoke(
        initial_state()
    )

    nodes.rewrite.assert_called_once()
    nodes.retrieve.assert_called_once()
    nodes.generate.assert_called_once()


# ---------------------------------------------------------
# State Propagation
# ---------------------------------------------------------


def test_graph_preserves_tenant_id_through_pipeline():
    nodes = build_mock_nodes()

    def rewrite(state):
        assert state["tenant_id"] == "tenant-a"

        return {
            **state,
            "rewritten_query": "rewritten",
        }

    def retrieve(state):
        assert state["tenant_id"] == "tenant-a"

        return {
            **state,
            "documents": [],
        }

    def generate(state):
        assert state["tenant_id"] == "tenant-a"

        return {
            **state,
            "answer": "Answer.",
            "claims": [],
            "citations": [],
        }

    nodes.rewrite.side_effect = rewrite
    nodes.retrieve.side_effect = retrieve
    nodes.generate.side_effect = generate

    graph = build_rag_graph(nodes)

    result = graph.invoke(
        initial_state()
    )

    assert result["tenant_id"] == "tenant-a"


def test_graph_preserves_question_through_pipeline():
    nodes = build_mock_nodes()

    question = (
        "What technologies did Shayan use?"
    )

    state = {
        **initial_state(),
        "question": question,
    }

    def rewrite(current_state):
        assert current_state["question"] == question

        return {
            **current_state,
            "rewritten_query": "rewritten",
        }

    def retrieve(current_state):
        assert current_state["question"] == question

        return {
            **current_state,
            "documents": [],
        }

    def generate(current_state):
        assert current_state["question"] == question

        return {
            **current_state,
            "answer": "Answer.",
            "claims": [],
            "citations": [],
        }

    nodes.rewrite.side_effect = rewrite
    nodes.retrieve.side_effect = retrieve
    nodes.generate.side_effect = generate

    graph = build_rag_graph(nodes)

    result = graph.invoke(state)

    assert result["question"] == question


def test_graph_preserves_chat_history():
    nodes = build_mock_nodes()

    chat_history = (
        "User asked about Shayan's frontend experience."
    )

    state = {
        **initial_state(),
        "chat_history": chat_history,
    }

    def rewrite(current_state):
        assert current_state["chat_history"] == chat_history

        return {
            **current_state,
            "rewritten_query": "rewritten",
        }

    def retrieve(current_state):
        assert current_state["chat_history"] == chat_history

        return {
            **current_state,
            "documents": [],
        }

    def generate(current_state):
        assert current_state["chat_history"] == chat_history

        return {
            **current_state,
            "answer": "Answer.",
            "claims": [],
            "citations": [],
        }

    nodes.rewrite.side_effect = rewrite
    nodes.retrieve.side_effect = retrieve
    nodes.generate.side_effect = generate

    graph = build_rag_graph(nodes)

    result = graph.invoke(state)

    assert result["chat_history"] == chat_history


def test_graph_preserves_user_memory():
    nodes = build_mock_nodes()

    user_memory = [
        {
            "key": "name",
            "value": "Shayan Javed",
            "memory_type": "profile",
        },
        {
            "key": "university",
            "value": "SMIU",
            "memory_type": "education",
        },
    ]

    state = {
        **initial_state(),
        "user_memory": user_memory,
    }

    graph = build_rag_graph(nodes)

    result = graph.invoke(state)

    assert result["user_memory"] == user_memory


# ---------------------------------------------------------
# Node Failure Propagation
# ---------------------------------------------------------


def test_graph_propagates_rewrite_failure():
    nodes = build_mock_nodes()

    nodes.rewrite.side_effect = RuntimeError(
        "Rewrite failed"
    )

    graph = build_rag_graph(nodes)

    try:
        graph.invoke(initial_state())
        assert False, "Expected rewrite failure"
    except RuntimeError as exc:
        assert str(exc) == "Rewrite failed"

    nodes.retrieve.assert_not_called()
    nodes.generate.assert_not_called()


def test_graph_propagates_retrieve_failure():
    nodes = build_mock_nodes()

    nodes.retrieve.side_effect = RuntimeError(
        "Retrieval failed"
    )

    graph = build_rag_graph(nodes)

    try:
        graph.invoke(initial_state())
        assert False, "Expected retrieval failure"
    except RuntimeError as exc:
        assert str(exc) == "Retrieval failed"

    nodes.rewrite.assert_called_once()
    nodes.generate.assert_not_called()


def test_graph_propagates_generate_failure():
    nodes = build_mock_nodes()

    nodes.generate.side_effect = RuntimeError(
        "Generation failed"
    )

    graph = build_rag_graph(nodes)

    try:
        graph.invoke(initial_state())
        assert False, "Expected generation failure"
    except RuntimeError as exc:
        assert str(exc) == "Generation failed"

    nodes.rewrite.assert_called_once()
    nodes.retrieve.assert_called_once()