from app.rag.vectorstore.qdrant import QdrantService

q = QdrantService()

offset = None
matches = []
total = 0

while True:
    points, next_offset = q.client.scroll(
        collection_name=q.collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    total += len(points)

    for p in points:
        payload = p.payload or {}

        if payload.get("file_name") == "bc2363a9-2d1d-467d-be92-ab96ab050a26.pdf":
            matches.append(p)

    if next_offset is None:
        break

    offset = next_offset

print("TOTAL POINTS SCANNED:", total)
print("OLD CV CHUNKS FOUND:", len(matches))

for p in matches:
    payload = p.payload or {}
    print(
        "ID:", p.id,
        "| chunk:", payload.get("chunk_index"),
        "| tenant:", payload.get("tenant_id"),
        "| section:", payload.get("section"),
    )
