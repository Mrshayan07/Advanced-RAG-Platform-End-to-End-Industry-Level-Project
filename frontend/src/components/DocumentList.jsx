
import { useEffect, useState } from "react";
import { deleteDocument, getDocuments } from "../services/documents";

export default function DocumentList({ refreshTrigger = 0 }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState("");

  const loadDocuments = async () => {
    try {
      setError("");

      const data = await getDocuments();
      setDocuments(data);
    } catch (err) {
      console.error("Failed to load documents:", err);
      setError(
        err.response?.data?.detail || "Failed to load documents."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, [refreshTrigger]);

  const handleDelete = async (documentId, documentName) => {
    const confirmed = window.confirm(
      `Delete "${documentName}"? This will remove the document from the knowledge base.`
    );

    if (!confirmed) {
      return;
    }

    try {
      setError("");
      setDeletingId(documentId);

      await deleteDocument(documentId);

      setDocuments((previous) =>
        previous.filter((document) => document.id !== documentId)
      );
    } catch (err) {
      console.error("Failed to delete document:", err);

      setError(
        err.response?.data?.detail || "Failed to delete document."
      );
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) {
    return (
      <div className="document-list">
        <div className="document-list-header">
          <h3>Knowledge Documents</h3>
        </div>

        <div className="document-list-status">
          Loading documents...
        </div>
      </div>
    );
  }

  return (
    <div className="document-list">
      <div className="document-list-header">
        <div>
          <h3>Knowledge Documents</h3>
          <p>
            {documents.length}{" "}
            {documents.length === 1 ? "document" : "documents"}
          </p>
        </div>
      </div>

      {error && (
        <div className="document-list-error">
          {error}
        </div>
      )}

      {documents.length === 0 ? (
        <div className="document-list-status">
          No documents uploaded yet.
        </div>
      ) : (
        <div className="document-items">
          {documents.map((document) => (
            <div
              key={document.id}
              className="document-item"
            >
              <div className="document-item-info">
                <div className="document-item-icon">
                  {document.mime_type === "application/pdf"
                    ? "PDF"
                    : "TXT"}
                </div>

                <div className="document-item-content">
                  <span
                    className="document-item-name"
                    title={document.name}
                  >
                    {document.name}
                  </span>

                  <span className="document-item-meta">
                    {document.status} •{" "}
                    {new Date(
                      document.created_at
                    ).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <button
                type="button"
                className="document-delete-button"
                onClick={() =>
                  handleDelete(
                    document.id,
                    document.name
                  )
                }
                disabled={deletingId === document.id}
                aria-label={`Delete ${document.name}`}
              >
                {deletingId === document.id ? "..." : "×"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
