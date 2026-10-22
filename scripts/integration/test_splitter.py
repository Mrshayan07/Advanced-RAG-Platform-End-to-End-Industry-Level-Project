from app.ingestion.loaders.pdf import load_pdf_file
from app.ingestion.processors.splitter import split_documents


pdf_path = r"storage\documents\bc2363a9-2d1d-467d-be92-ab96ab050a26.pdf"

documents = load_pdf_file(pdf_path)
chunks = split_documents(documents)

print(f"Chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print("\n" + "=" * 70)
    print(f"CHUNK {index}")
    print("=" * 70)

    print(f"Section: {chunk.metadata.get('section', 'N/A')}")

    print("\nContent:")
    print(chunk.page_content)