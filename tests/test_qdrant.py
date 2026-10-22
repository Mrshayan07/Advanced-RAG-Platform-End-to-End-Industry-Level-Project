from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.rag.vectorstore.qdrant import QdrantService

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def build_service():
    """
    Build QdrantService without creating a real Qdrant client.
    """

    service = QdrantService.__new__(QdrantService)

    service.client = MagicMock()
    service.collection_name = "test_collection"

    return service


def make_payload(
    tenant_id="tenant-a",
    source="resume.pdf",
    chunk_index=0,
    text="Shayan is a Full Stack Developer.",
):
    return {
        "tenant_id": tenant_id,
        "source": source,
        "chunk_index": chunk_index,
        "text": text,
    }


# ---------------------------------------------------------
# Collection Tests
# ---------------------------------------------------------


def test_create_collection_creates_missing_collection():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[]
    )

    service.create_collection()

    service.client.create_collection.assert_called_once_with(
        collection_name="test_collection",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


def test_create_collection_does_not_recreate_existing_collection():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[
            SimpleNamespace(
                name="test_collection"
            )
        ]
    )

    service.create_collection()

    service.client.create_collection.assert_not_called()


def test_create_collection_creates_tenant_payload_index():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[]
    )

    service.create_collection()

    calls = (
        service.client.create_payload_index.call_args_list
    )

    assert len(calls) == 2

    tenant_call = calls[0]

    assert tenant_call.kwargs[
        "collection_name"
    ] == "test_collection"

    assert tenant_call.kwargs[
        "field_name"
    ] == "tenant_id"

    assert tenant_call.kwargs[
        "field_schema"
    ] == PayloadSchemaType.KEYWORD


def test_create_collection_creates_source_payload_index():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[]
    )

    service.create_collection()

    calls = (
        service.client.create_payload_index.call_args_list
    )

    assert len(calls) == 2

    source_call = calls[1]

    assert source_call.kwargs[
        "collection_name"
    ] == "test_collection"

    assert source_call.kwargs[
        "field_name"
    ] == "source"

    assert source_call.kwargs[
        "field_schema"
    ] == PayloadSchemaType.KEYWORD


def test_create_collection_ignores_existing_payload_indexes():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[]
    )

    service.client.create_payload_index.side_effect = [
        Exception("Index already exists"),
        Exception("already exists"),
    ]

    service.create_collection()

    assert (
        service.client.create_payload_index.call_count
        == 2
    )


def test_create_collection_reraises_unexpected_payload_index_error():
    service = build_service()

    service.client.get_collections.return_value = SimpleNamespace(
        collections=[]
    )

    service.client.create_payload_index.side_effect = (
        RuntimeError("Qdrant connection failed")
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant connection failed",
    ):
        service.create_collection()


# ---------------------------------------------------------
# Point ID Tests
# ---------------------------------------------------------


def test_generate_point_id_is_deterministic():
    service = build_service()

    payload = make_payload(
        tenant_id="tenant-a",
        source="resume.pdf",
        chunk_index=3,
    )

    point_id_one = service._generate_point_id(
        payload
    )

    point_id_two = service._generate_point_id(
        payload
    )

    assert point_id_one == point_id_two


def test_generate_point_id_is_uuid_string():
    service = build_service()

    point_id = service._generate_point_id(
        make_payload()
    )

    assert isinstance(point_id, str)

    assert len(point_id) == 36

    assert point_id.count("-") == 4


def test_generate_point_id_changes_for_different_tenants():
    service = build_service()

    tenant_a_id = service._generate_point_id(
        make_payload(
            tenant_id="tenant-a"
        )
    )

    tenant_b_id = service._generate_point_id(
        make_payload(
            tenant_id="tenant-b"
        )
    )

    assert tenant_a_id != tenant_b_id


def test_generate_point_id_changes_for_different_sources():
    service = build_service()

    source_a_id = service._generate_point_id(
        make_payload(
            source="resume.pdf"
        )
    )

    source_b_id = service._generate_point_id(
        make_payload(
            source="projects.pdf"
        )
    )

    assert source_a_id != source_b_id


def test_generate_point_id_changes_for_different_chunks():
    service = build_service()

    chunk_zero_id = service._generate_point_id(
        make_payload(
            chunk_index=0
        )
    )

    chunk_one_id = service._generate_point_id(
        make_payload(
            chunk_index=1
        )
    )

    assert chunk_zero_id != chunk_one_id


def test_generate_point_id_requires_tenant_id():
    service = build_service()

    payload = {
        "source": "resume.pdf",
        "chunk_index": 0,
    }

    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        service._generate_point_id(payload)


# ---------------------------------------------------------
# Vector Upsert Tests
# ---------------------------------------------------------


def test_upsert_vectors_rejects_mismatched_lengths():
    service = build_service()

    vectors = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    payloads = [
        make_payload()
    ]

    with pytest.raises(
        ValueError,
        match="Number of vectors must match",
    ):
        service.upsert_vectors(
            vectors=vectors,
            payloads=payloads,
        )


def test_upsert_vectors_rejects_missing_tenant_id():
    service = build_service()

    vectors = [
        [0.1, 0.2]
    ]

    payloads = [
        {
            "source": "resume.pdf",
            "chunk_index": 0,
        }
    ]

    with pytest.raises(
        ValueError,
        match="Every Qdrant payload must contain tenant_id",
    ):
        service.upsert_vectors(
            vectors=vectors,
            payloads=payloads,
        )


def test_upsert_vectors_empty_lists_do_not_call_qdrant():
    service = build_service()

    service.upsert_vectors(
        vectors=[],
        payloads=[],
    )

    service.client.upsert.assert_not_called()


def test_upsert_vectors_creates_point_structs():
    service = build_service()

    vectors = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    payloads = [
        make_payload(
            chunk_index=0
        ),
        make_payload(
            chunk_index=1
        ),
    ]

    service.upsert_vectors(
        vectors=vectors,
        payloads=payloads,
    )

    service.client.upsert.assert_called_once()

    call = service.client.upsert.call_args

    assert call.kwargs[
        "collection_name"
    ] == "test_collection"

    points = call.kwargs["points"]

    assert len(points) == 2

    assert all(
        isinstance(point, PointStruct)
        for point in points
    )


def test_upsert_vectors_preserves_vector_and_payload():
    service = build_service()

    vector = [0.1, 0.2, 0.3]

    payload = make_payload()

    service.upsert_vectors(
        vectors=[vector],
        payloads=[payload],
    )

    points = (
        service.client.upsert.call_args.kwargs[
            "points"
        ]
    )

    point = points[0]

    assert point.vector == vector

    assert point.payload == payload


# ---------------------------------------------------------
# Document Upsert Tests
# ---------------------------------------------------------


def test_upsert_documents_embeds_document_text():
    service = build_service()

    embedding_service = MagicMock()

    embedding_service.embed_documents.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    documents = [
        Document(
            page_content="First document.",
            metadata={
                "source": "first.pdf",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Second document.",
            metadata={
                "source": "second.pdf",
                "chunk_index": 1,
            },
        ),
    ]

    service.upsert_vectors = MagicMock()

    service.upsert_documents(
        documents=documents,
        embedding_service=embedding_service,
        tenant_id="tenant-a",
    )

    embedding_service.embed_documents.assert_called_once_with(
        [
            "First document.",
            "Second document.",
        ]
    )


def test_upsert_documents_builds_tenant_aware_payloads():
    service = build_service()

    embedding_service = MagicMock()

    embedding_service.embed_documents.return_value = [
        [0.1, 0.2]
    ]

    documents = [
        Document(
            page_content="Shayan uses FastAPI.",
            metadata={
                "source": "resume.pdf",
                "chunk_index": 0,
            },
        )
    ]

    service.upsert_vectors = MagicMock()

    service.upsert_documents(
        documents=documents,
        embedding_service=embedding_service,
        tenant_id="tenant-a",
    )

    service.upsert_vectors.assert_called_once()

    call = service.upsert_vectors.call_args

    assert call.kwargs["vectors"] == [
        [0.1, 0.2]
    ]

    assert call.kwargs["payloads"] == [
        {
            "text": "Shayan uses FastAPI.",
            "source": "resume.pdf",
            "chunk_index": 0,
            "tenant_id": "tenant-a",
        }
    ]


def test_upsert_documents_normalizes_uuid_tenant_id():
    service = build_service()

    embedding_service = MagicMock()

    embedding_service.embed_documents.return_value = [
        [0.1, 0.2]
    ]

    documents = [
        Document(
            page_content="Document text.",
            metadata={
                "source": "resume.pdf",
            },
        )
    ]

    service.upsert_vectors = MagicMock()

    tenant_id = "550e8400-e29b-41d4-a716-446655440000"

    service.upsert_documents(
        documents=documents,
        embedding_service=embedding_service,
        tenant_id=tenant_id,
    )

    payloads = (
        service.upsert_vectors.call_args.kwargs[
            "payloads"
        ]
    )

    assert payloads[0]["tenant_id"] == tenant_id


# ---------------------------------------------------------
# Tenant Filter Tests
# ---------------------------------------------------------


def test_tenant_filter_contains_tenant_id():
    tenant_filter = QdrantService._tenant_filter(
        "tenant-a"
    )

    assert len(
        tenant_filter.must
    ) == 1

    condition = tenant_filter.must[0]

    assert condition.key == "tenant_id"

    assert condition.match.value == "tenant-a"


def test_tenant_filter_normalizes_tenant_id():
    tenant_filter = QdrantService._tenant_filter(
        123
    )

    condition = tenant_filter.must[0]

    assert condition.match.value == "123"


def test_tenant_filter_rejects_empty_tenant_id():
    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        QdrantService._tenant_filter("")


# ---------------------------------------------------------
# Search Tests
# ---------------------------------------------------------


def test_search_requires_tenant_id():
    service = build_service()

    with pytest.raises(
        ValueError,
        match="tenant_id is required",
    ):
        service.search(
            query_vector=[0.1, 0.2],
            limit=5,
            tenant_id=None,
        )


def test_search_applies_tenant_filter():
    service = build_service()

    service.client.query_points.return_value = (
        SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    payload={
                        "tenant_id": "tenant-a",
                        "text": "Document A",
                    },
                )
            ]
        )
    )

    results = service.search(
        query_vector=[0.1, 0.2],
        limit=3,
        tenant_id="tenant-a",
    )

    service.client.query_points.assert_called_once()

    call = service.client.query_points.call_args

    assert call.kwargs[
        "collection_name"
    ] == "test_collection"

    assert call.kwargs[
        "query"
    ] == [0.1, 0.2]

    assert call.kwargs[
        "limit"
    ] == 3

    assert call.kwargs[
        "with_payload"
    ] is True

    query_filter = call.kwargs[
        "query_filter"
    ]

    assert query_filter.must[0].key == "tenant_id"

    assert (
        query_filter.must[0].match.value
        == "tenant-a"
    )

    assert len(results) == 1


def test_search_returns_qdrant_points():
    service = build_service()

    points = [
        SimpleNamespace(
            id="point-1",
            payload={
                "tenant_id": "tenant-a",
                "text": "Document A",
            },
        ),
        SimpleNamespace(
            id="point-2",
            payload={
                "tenant_id": "tenant-a",
                "text": "Document B",
            },
        ),
    ]

    service.client.query_points.return_value = (
        SimpleNamespace(
            points=points
        )
    )

    result = service.search(
        query_vector=[0.1, 0.2],
        limit=5,
        tenant_id="tenant-a",
    )

    assert result == points


# ---------------------------------------------------------
# Get All Documents Tests
# ---------------------------------------------------------


def test_get_all_documents_returns_documents_from_qdrant():
    service = build_service()

    service.client.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "text": "Shayan uses React.",
                    "tenant_id": "tenant-a",
                    "source": "resume.pdf",
                    "chunk_index": 0,
                }
            )
        ],
        None,
    )

    documents = service.get_all_documents(
        batch_size=100,
        tenant_id="tenant-a",
    )

    assert len(documents) == 1

    assert documents[0].page_content == (
        "Shayan uses React."
    )

    assert documents[0].metadata == {
        "tenant_id": "tenant-a",
        "source": "resume.pdf",
        "chunk_index": 0,
    }


def test_get_all_documents_excludes_text_from_metadata():
    service = build_service()

    service.client.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "text": "Document text.",
                    "tenant_id": "tenant-a",
                    "source": "resume.pdf",
                }
            )
        ],
        None,
    )

    documents = service.get_all_documents(
        tenant_id="tenant-a"
    )

    assert "text" not in documents[0].metadata


def test_get_all_documents_skips_empty_text_payloads():
    service = build_service()

    service.client.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "text": "",
                    "tenant_id": "tenant-a",
                }
            ),
            SimpleNamespace(
                payload={
                    "tenant_id": "tenant-a",
                }
            ),
            SimpleNamespace(
                payload={
                    "text": "Valid document.",
                    "tenant_id": "tenant-a",
                }
            ),
        ],
        None,
    )

    documents = service.get_all_documents(
        tenant_id="tenant-a"
    )

    assert len(documents) == 1

    assert documents[0].page_content == (
        "Valid document."
    )


def test_get_all_documents_uses_tenant_filter():
    service = build_service()

    service.client.scroll.return_value = (
        [],
        None,
    )

    service.get_all_documents(
        tenant_id="tenant-a"
    )

    call = service.client.scroll.call_args

    query_filter = call.kwargs[
        "scroll_filter"
    ]

    assert query_filter.must[0].key == "tenant_id"

    assert (
        query_filter.must[0].match.value
        == "tenant-a"
    )


def test_get_all_documents_without_tenant_has_no_filter():
    service = build_service()

    service.client.scroll.return_value = (
        [],
        None,
    )

    service.get_all_documents()

    call = service.client.scroll.call_args

    assert call.kwargs[
        "scroll_filter"
    ] is None


def test_get_all_documents_handles_pagination():
    service = build_service()

    first_page = [
        SimpleNamespace(
            payload={
                "text": "First page.",
                "tenant_id": "tenant-a",
            }
        )
    ]

    second_page = [
        SimpleNamespace(
            payload={
                "text": "Second page.",
                "tenant_id": "tenant-a",
            }
        )
    ]

    service.client.scroll.side_effect = [
        (
            first_page,
            "offset-1",
        ),
        (
            second_page,
            None,
        ),
    ]

    documents = service.get_all_documents(
        batch_size=1,
        tenant_id="tenant-a",
    )

    assert len(documents) == 2

    assert documents[0].page_content == (
        "First page."
    )

    assert documents[1].page_content == (
        "Second page."
    )

    assert (
        service.client.scroll.call_count
        == 2
    )

    second_call = (
        service.client.scroll.call_args_list[1]
    )

    assert second_call.kwargs[
        "offset"
    ] == "offset-1"


# ---------------------------------------------------------
# Delete Tests
# ---------------------------------------------------------


def test_delete_documents_by_source_requires_source():
    service = build_service()

    with pytest.raises(
        ValueError,
        match="source is required",
    ):
        service.delete_documents_by_source(
            source="",
            tenant_id="tenant-a",
        )


def test_delete_documents_by_source_applies_tenant_and_source_filters():
    service = build_service()

    service.client.scroll.return_value = (
        [
            SimpleNamespace(id="point-1"),
            SimpleNamespace(id="point-2"),
        ],
        None,
    )

    service.delete_documents_by_source(
        source="resume.pdf",
        tenant_id="tenant-a",
    )

    service.client.scroll.assert_called_once()

    call = service.client.scroll.call_args

    query_filter = call.kwargs[
        "scroll_filter"
    ]

    assert len(query_filter.must) == 2

    assert (
        query_filter.must[0].key
        == "tenant_id"
    )

    assert (
        query_filter.must[0].match.value
        == "tenant-a"
    )

    assert (
        query_filter.must[1].key
        == "source"
    )

    assert (
        query_filter.must[1].match.value
        == "resume.pdf"
    )


def test_delete_documents_by_source_deletes_found_points():
    service = build_service()

    service.client.scroll.return_value = (
        [
            SimpleNamespace(id="point-1"),
            SimpleNamespace(id="point-2"),
        ],
        None,
    )

    service.delete_documents_by_source(
        source="resume.pdf",
        tenant_id="tenant-a",
    )

    service.client.delete.assert_called_once()

    call = service.client.delete.call_args

    assert call.kwargs[
        "collection_name"
    ] == "test_collection"

    assert call.kwargs[
        "points_selector"
    ] == [
        "point-1",
        "point-2",
    ]


def test_delete_documents_by_source_does_nothing_when_no_points_found():
    service = build_service()

    service.client.scroll.return_value = (
        [],
        None,
    )

    service.delete_documents_by_source(
        source="missing.pdf",
        tenant_id="tenant-a",
    )

    service.client.delete.assert_not_called()


def test_delete_documents_by_source_handles_pagination():
    service = build_service()

    service.client.scroll.side_effect = [
        (
            [
                SimpleNamespace(
                    id="point-1"
                )
            ],
            "offset-1",
        ),
        (
            [
                SimpleNamespace(
                    id="point-2"
                )
            ],
            None,
        ),
    ]

    service.delete_documents_by_source(
        source="resume.pdf",
        tenant_id="tenant-a",
    )

    assert (
        service.client.scroll.call_count
        == 2
    )

    service.client.delete.assert_called_once()

    deleted_ids = (
        service.client.delete.call_args.kwargs[
            "points_selector"
        ]
    )

    assert deleted_ids == [
        "point-1",
        "point-2",
    ]