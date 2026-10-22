from __future__ import annotations

from collections import defaultdict
from threading import Lock
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from langsmith import traceable
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import Conversation, Message
from app.rag.embeddings.service import EmbeddingService
from app.rag.graph.nodes import RAGNodes
from app.rag.graph.workflow import build_rag_graph
from app.rag.query.rewriting import QueryRewriter
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.bm25 import BM25Service
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.vectorstore.qdrant import QdrantService
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.memory import ConversationMemory
from app.services.memory_manager import MemoryManager
from app.services.user_memory import UserMemoryService

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


# ============================================================
# RAG PIPELINE
# ============================================================

def build_rag_pipeline():
    """
    Build the complete production-oriented RAG pipeline.

    Components:

        Query rewriting
        +
        Tenant-aware BM25
        +
        Dense vector retrieval
        +
        RRF fusion
        +
        Cross-encoder reranking
        +
        LangGraph generation

    Heavy model initialization intentionally happens here,
    not during module import.
    """

    # ========================================================
    # QUERY REWRITER
    # ========================================================

    query_rewriter = QueryRewriter()

    # ========================================================
    # BM25
    # ========================================================

    bm25_service = BM25Service()

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embedding_service = EmbeddingService()

    # ========================================================
    # QDRANT
    # ========================================================

    qdrant_service = QdrantService(
    )

    # ========================================================
    # LOAD EXISTING DOCUMENTS
    # ========================================================

    documents = qdrant_service.get_all_documents()

    tenant_documents: dict[str, list] = defaultdict(list)

    for document in documents:
        tenant_id = document.metadata.get("tenant_id")

        if not tenant_id:
            continue

        tenant_documents[str(tenant_id)].append(document)

    # ========================================================
    # BUILD TENANT-SPECIFIC BM25 INDEXES
    # ========================================================

    for tenant_id, tenant_docs in tenant_documents.items():
        if not tenant_docs:
            continue

        try:
            bm25_service.build_index(
                documents=tenant_docs,
                tenant_id=tenant_id,
            )
        except Exception as exc:
            print(
                "[RAG] Failed to build BM25 index "
                f"for tenant={tenant_id}: {exc}"
            )

    if not tenant_documents:
        print(
            "[RAG] No tenant documents available "
            "for initial BM25 indexing."
        )

    # ========================================================
    # CROSS-ENCODER RERANKER
    # ========================================================

    reranker = CrossEncoderReranker()

    # ========================================================
    # HYBRID RETRIEVER
    # ========================================================

    hybrid_retriever = HybridRetriever(
        bm25_service=bm25_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    # ========================================================
    # RAG NODES
    # ========================================================

    nodes = RAGNodes(
        query_rewriter=query_rewriter,
        retriever=hybrid_retriever,
    )

    # ========================================================
    # LANGGRAPH
    # ========================================================

    return build_rag_graph(nodes)


# ============================================================
# LAZY RAG INITIALIZATION
# ============================================================

rag_graph = None
rag_graph_lock = Lock()


def get_rag_graph():
    """
    Return the shared RAG graph.

    Heavy RAG components are initialized only when the
    first RAG request arrives.

    A process-local lock prevents multiple concurrent
    requests from initializing the pipeline simultaneously.
    """

    global rag_graph

    if rag_graph is None:
        with rag_graph_lock:
            if rag_graph is None:
                print(
                    "[RAG] Initializing RAG pipeline..."
                )

                rag_graph = build_rag_pipeline()

                print(
                    "[RAG] RAG pipeline initialization complete."
                )

    return rag_graph


# ============================================================
# REDIS CONVERSATION MEMORY
# ============================================================

conversation_memory = ConversationMemory()


# ============================================================
# CONVERSATION HELPERS
# ============================================================

async def get_or_create_conversation(
    *,
    db: AsyncSession,
    user_id: UUID,
    tenant_id: UUID,
    conversation_id: UUID | None,
) -> Conversation:
    """
    Retrieve an existing conversation belonging to the
    authenticated user and tenant, or create a new one.

    Tenant and user ownership are always enforced.
    """

    if conversation_id is not None:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.tenant_id == tenant_id,
            )
        )

        conversation = result.scalar_one_or_none()

        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

        return conversation

    conversation = Conversation(
        id=uuid4(),
        user_id=user_id,
        tenant_id=tenant_id,
        title="New Conversation",
    )

    db.add(conversation)

    await db.commit()
    await db.refresh(conversation)

    return conversation


async def update_conversation_title(
    *,
    db: AsyncSession,
    conversation: Conversation,
    question: str,
) -> None:
    """
    Assign a lightweight title to a newly-created conversation.

    Existing meaningful titles are not overwritten.
    """

    if conversation.title and conversation.title != "New Conversation":
        return

    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        return

    conversation.title = normalized_question[:80]

    await db.commit()
    await db.refresh(conversation)


# ============================================================
# POSTGRES CONVERSATION PERSISTENCE
# ============================================================

async def save_postgres_turn(
    *,
    db: AsyncSession,
    conversation_id: UUID,
    question: str,
    answer: str,
    citations: list | None = None,
) -> None:
    """
    Persist one complete RAG turn in PostgreSQL.

    User messages are stored with an empty citations list.
    Assistant messages store the citations returned by the
    RAG pipeline.

    Conversation ownership is already enforced by
    get_or_create_conversation().
    """

    # --------------------------------------------------------
    # Convert Pydantic/model objects into JSON-compatible data
    # for the PostgreSQL JSONB column.
    # --------------------------------------------------------

    conversation_result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
        )
        .with_for_update()
    )

    conversation = conversation_result.scalar_one_or_none()

    if conversation is None:
        raise ValueError(
            f"Conversation {conversation_id} does not exist."
        )

    max_sequence_result = await db.execute(
        select(
            func.coalesce(
                func.max(Message.sequence_number),
                0,
            )
        ).where(
            Message.conversation_id == conversation_id,
        )
    )

    current_max_sequence = max_sequence_result.scalar_one()

    user_sequence = current_max_sequence + 1
    assistant_sequence = current_max_sequence + 2

    serialized_citations = jsonable_encoder(
        citations or []
    )

    user_message = Message(
        conversation_id=conversation_id,
        sequence_number=user_sequence,
        role="user",
        content=question,
        citations=[],
    )

    assistant_message = Message(
        conversation_id=conversation_id,
        sequence_number=assistant_sequence,
        role="assistant",
        content=answer,
        citations=serialized_citations,
    )

    db.add(user_message)
    db.add(assistant_message)

    await db.commit()

# ============================================================
# LOAD DURABLE USER MEMORY
# ============================================================

async def load_user_memory(
    *,
    db: AsyncSession,
    user_id: UUID,
    tenant_id: UUID,
) -> list[dict[str, str]]:
    """
    Load durable user memories from PostgreSQL.

    Memory is explicitly filtered by both tenant_id and
    user_id to maintain multi-tenant isolation.
    """

    memory_service = UserMemoryService(db)

    memories = await memory_service.get_memories(
        tenant_id=tenant_id,
        user_id=user_id,
    )

    return [
        {
            "key": memory.memory_key,
            "value": memory.memory_value,
            "memory_type": memory.memory_type,
        }
        for memory in memories
    ]


# ============================================================
# LANGGRAPH EXECUTION
# ============================================================

@traceable(
    name="RAG API Request",
    run_type="chain",
)
def run_rag_graph(
    *,
    question: str,
    chat_history: str,
    user_memory: list[dict[str, str]],
    tenant_id: str,
    user_id: str,
    role: str,
):
    """
    Execute the shared LangGraph RAG pipeline.

    Tenant/user/role are passed into graph state and
    LangSmith metadata for observability and isolation.
    """

    graph = get_rag_graph()

    return graph.invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "user_memory": user_memory,
            "tenant_id": tenant_id,
        },
        config={
            "tags": [
                "rag-api",
                "production-rag",
            ],
            "metadata": {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role,
            },
        },
    )


# ============================================================
# RAG QUERY ENDPOINT
# ============================================================

@router.post(
    "/query",
    response_model=RAGQueryResponse,
)
async def query_rag(
    data: RAGQueryRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a tenant-isolated RAG query.

    Authentication:
        JWT
          ↓
        user_id
        tenant_id
        role

    Memory:
        Redis short-term conversation memory
        +
        PostgreSQL durable user memory

    RAG:
        Query rewriting
        +
        Hybrid retrieval
        +
        RRF
        +
        Cross-encoder reranking
        +
        LangGraph generation
    """

    user_id = current_user["user_id"]
    tenant_id = current_user["tenant_id"]
    role = current_user["role"]

    # ========================================================
    # CONVERSATION
    # ========================================================

    conversation = await get_or_create_conversation(
        db=db,
        user_id=user_id,
        tenant_id=tenant_id,
        conversation_id=data.conversation_id,
    )

    # ========================================================
    # REDIS SHORT-TERM MEMORY
    # ========================================================

    chat_history = conversation_memory.build_chat_history(
        conversation_id=str(conversation.id),
    )

    # ========================================================
    # POSTGRES DURABLE USER MEMORY
    # ========================================================

    user_memory = await load_user_memory(
        db=db,
        user_id=user_id,
        tenant_id=tenant_id,
    )

    # ========================================================
    # DURABLE MEMORY MANAGER
    # ========================================================

    memory_manager = MemoryManager(
        db=db,
    )

    # ========================================================
    # RUN RAG
    # ========================================================

    result = run_rag_graph(
        question=data.question,
        chat_history=chat_history,
        user_memory=user_memory,
        tenant_id=str(tenant_id),
        user_id=str(user_id),
        role=role,
    )

    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = result.get(
        "answer",
        "",
    )

    citations = result.get(
        "citations",
        [],
    )

    # ========================================================
    # SAVE REDIS SHORT-TERM MEMORY
    # ========================================================

    conversation_memory.save_turn(
        conversation_id=str(conversation.id),
        question=data.question,
        answer=answer,
    )

    # ========================================================
    # SAVE POSTGRES CONVERSATION
    # ========================================================

    await save_postgres_turn(
        db=db,
        conversation_id=conversation.id,
        question=data.question,
        answer=answer,
        citations=citations,
    )

    # ========================================================
    # UPDATE TITLE
    # ========================================================

    await update_conversation_title(
        db=db,
        conversation=conversation,
        question=data.question,
    )

    # ========================================================
    # EXTRACT DURABLE USER MEMORY
    # ========================================================

    await memory_manager.process_user_message(
        tenant_id=tenant_id,
        user_id=user_id,
        user_message=data.question,
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return RAGQueryResponse(
        conversation_id=str(conversation.id),
        answer=answer,
        citations=citations,
    )