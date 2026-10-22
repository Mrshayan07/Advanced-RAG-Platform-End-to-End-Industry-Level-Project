from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_file(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents