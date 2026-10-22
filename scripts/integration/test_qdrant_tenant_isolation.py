from __future__ import annotations

import uuid

from app.rag.vectorstore.qdrant import QdrantService


VECTOR_SIZE = 384


def make_vector(value: float) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[0] = value
    return vector


def main() -> None:
    qdrant = QdrantService()

    original_collection = qdrant.collection_name
    temporary_collection = (
        f"advanced_rag_tenant_test_{uuid.uuid4().hex[:12]}"
    )

    tenant_a = str(uuid.uuid4())
    tenant_b = str(uuid.uuid4())
    unknown_tenant = str(uuid.uuid4())

    print("=" * 70)
    print("QDRANT MULTI-TENANT ISOLATION TEST")
    print("=" * 70)

    print(f"\nTemporary collection: {temporary_collection}")
    print(f"Tenant A: {tenant_a}")
    print(f"Tenant B: {tenant_b}")

    try:
        # ---------------------------------------------------------
        # 1. Use isolated temporary collection
        # ---------------------------------------------------------
        qdrant.collection_name = temporary_collection
        qdrant.create_collection()

        print("\n[PASS] Temporary collection created.")

        # ---------------------------------------------------------
        # 2. Prepare documents
        #
        # Same source + chunk_index is intentionally used for
        # both tenants.
        #
        # This verifies that tenant_id participates in point ID
        # generation and prevents collisions.
        # ---------------------------------------------------------
        tenant_a_payloads = [
            {
                "text": "Tenant A secret document",
                "tenant_id": tenant_a,
                "source": "shared.txt",
                "chunk_index": 0,
            },
            {
                "text": "Tenant A private information",
                "tenant_id": tenant_a,
                "source": "private-a.txt",
                "chunk_index": 1,
            },
        ]

        tenant_b_payloads = [
            {
                "text": "Tenant B secret document",
                "tenant_id": tenant_b,
                "source": "shared.txt",
                "chunk_index": 0,
            },
            {
                "text": "Tenant B private information",
                "tenant_id": tenant_b,
                "source": "private-b.txt",
                "chunk_index": 1,
            },
        ]

        vectors = [
            make_vector(1.0),
            make_vector(0.9),
            make_vector(1.0),
            make_vector(0.9),
        ]

        payloads = (
            tenant_a_payloads
            + tenant_b_payloads
        )

        # ---------------------------------------------------------
        # 3. Verify tenant-specific point IDs
        # ---------------------------------------------------------
        tenant_a_id = qdrant._generate_point_id(
            tenant_a_payloads[0]
        )

        tenant_b_id = qdrant._generate_point_id(
            tenant_b_payloads[0]
        )

        assert tenant_a_id != tenant_b_id, (
            "Point IDs must differ across tenants."
        )

        print(
            "[PASS] Same source/chunk gets different point IDs "
            "across tenants."
        )

        # ---------------------------------------------------------
        # 4. Verify missing tenant_id is rejected
        # ---------------------------------------------------------
        try:
            qdrant.upsert_vectors(
                vectors=[make_vector(1.0)],
                payloads=[
                    {
                        "text": "Invalid document",
                        "source": "invalid.txt",
                        "chunk_index": 0,
                    }
                ],
            )

            raise AssertionError(
                "Missing tenant_id was not rejected."
            )

        except ValueError as exc:
            assert "tenant_id" in str(exc).lower()
            print(
                "[PASS] Missing tenant_id correctly rejected."
            )

        # ---------------------------------------------------------
        # 5. Insert Tenant A + Tenant B documents
        # ---------------------------------------------------------
        qdrant.upsert_vectors(
            vectors=vectors,
            payloads=payloads,
        )

        print(
            "[PASS] Tenant A and Tenant B documents indexed."
        )

        # ---------------------------------------------------------
        # 6. Search as Tenant A
        # ---------------------------------------------------------
        tenant_a_results = qdrant.search(
            query_vector=make_vector(1.0),
            limit=10,
            tenant_id=tenant_a,
        )

        assert tenant_a_results, (
            "Tenant A should receive search results."
        )

        tenant_a_result_tenants = {
            result.payload.get("tenant_id")
            for result in tenant_a_results
            if result.payload
        }

        assert tenant_a_result_tenants == {tenant_a}, (
            "Tenant A received documents belonging to "
            "another tenant."
        )

        tenant_a_texts = {
            result.payload.get("text")
            for result in tenant_a_results
            if result.payload
        }

        assert "Tenant A secret document" in tenant_a_texts
        assert "Tenant A private information" in tenant_a_texts

        assert "Tenant B secret document" not in tenant_a_texts
        assert "Tenant B private information" not in tenant_a_texts

        print(
            "[PASS] Tenant A can retrieve only Tenant A documents."
        )

        # ---------------------------------------------------------
        # 7. Search as Tenant B
        # ---------------------------------------------------------
        tenant_b_results = qdrant.search(
            query_vector=make_vector(1.0),
            limit=10,
            tenant_id=tenant_b,
        )

        assert tenant_b_results, (
            "Tenant B should receive search results."
        )

        tenant_b_result_tenants = {
            result.payload.get("tenant_id")
            for result in tenant_b_results
            if result.payload
        }

        assert tenant_b_result_tenants == {tenant_b}, (
            "Tenant B received documents belonging to "
            "another tenant."
        )

        tenant_b_texts = {
            result.payload.get("text")
            for result in tenant_b_results
            if result.payload
        }

        assert "Tenant B secret document" in tenant_b_texts
        assert "Tenant B private information" in tenant_b_texts

        assert "Tenant A secret document" not in tenant_b_texts
        assert "Tenant A private information" not in tenant_b_texts

        print(
            "[PASS] Tenant B can retrieve only Tenant B documents."
        )

        # ---------------------------------------------------------
        # 8. Unknown tenant must receive zero results
        # ---------------------------------------------------------
        unknown_results = qdrant.search(
            query_vector=make_vector(1.0),
            limit=10,
            tenant_id=unknown_tenant,
        )

        assert unknown_results == [], (
            "Unknown tenant should receive zero documents."
        )

        print(
            "[PASS] Unknown tenant receives zero documents."
        )

        # ---------------------------------------------------------
        # 9. Verify get_all_documents tenant filtering
        # ---------------------------------------------------------
        tenant_a_documents = qdrant.get_all_documents(
            tenant_id=tenant_a,
        )

        tenant_b_documents = qdrant.get_all_documents(
            tenant_id=tenant_b,
        )

        assert tenant_a_documents
        assert tenant_b_documents

        assert all(
            document.metadata.get("tenant_id") == tenant_a
            for document in tenant_a_documents
        ), "get_all_documents leaked Tenant B data to Tenant A."

        assert all(
            document.metadata.get("tenant_id") == tenant_b
            for document in tenant_b_documents
        ), "get_all_documents leaked Tenant A data to Tenant B."

        print(
            "[PASS] get_all_documents() respects tenant filtering."
        )

        # ---------------------------------------------------------
        # 10. Final success
        # ---------------------------------------------------------
        print("\n" + "=" * 70)
        print("ALL QDRANT TENANT ISOLATION TESTS PASSED")
        print("=" * 70)

    finally:
        # ---------------------------------------------------------
        # Cleanup temporary collection
        # ---------------------------------------------------------
        try:
            qdrant.client.delete_collection(
                collection_name=temporary_collection,
            )
            print(
                f"\n[CLEANUP] Deleted temporary collection: "
                f"{temporary_collection}"
            )
        except Exception as exc:
            print(
                f"\n[CLEANUP WARNING] Could not delete temporary "
                f"collection: {exc}"
            )

        qdrant.collection_name = original_collection


if __name__ == "__main__":
    main()