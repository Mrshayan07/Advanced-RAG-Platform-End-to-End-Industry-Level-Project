
import api from "./api";

export async function uploadDocument({
  file,
  chunkSize = 600,
  chunkOverlap = 150,
}) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("chunk_size", String(chunkSize));
  formData.append("chunk_overlap", String(chunkOverlap));

  const response = await api.post("/documents/upload", formData);

  return response.data;
}

export async function getDocuments() {
  const response = await api.get("/documents");

  return response.data;
}

export async function deleteDocument(documentId) {
  await api.delete(`/documents/${documentId}`);
}
