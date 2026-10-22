from langgraph.graph import END, START, StateGraph

from app.rag.graph.nodes import RAGNodes
from app.rag.graph.state import RAGState


def build_rag_graph(nodes: RAGNodes):
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()