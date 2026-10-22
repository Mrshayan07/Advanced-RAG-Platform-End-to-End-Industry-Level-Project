import { useState } from "react";
import { uploadDocument } from "../services/documents";

export default function DocumentUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [chunkSize, setChunkSize] = useState(600);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    setMessage("");
    setError("");

    if (!selectedFile) {
      setFile(null);
      return;
    }

    if (selectedFile.type !== "application/pdf") {
      setFile(null);
      setError("Please select a PDF file.");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    setMessage("");
    setError("");

    if (!file) {
      setError("Please select a PDF file.");
      return;
    }

    const size = Number(chunkSize);
    const overlap = Number(chunkOverlap);

    if (size < 300 || size > 2000) {
      setError("Chunk size must be between 300 and 2000.");
      return;
    }

    if (overlap < 0 || overlap > 500) {
      setError("Chunk overlap must be between 0 and 500.");
      return;
    }

    if (overlap >= size) {
      setError("Chunk overlap must be smaller than chunk size.");
      return;
    }

    try {
      setUploading(true);

      const document = await uploadDocument({
        file,
        chunkSize: size,
        chunkOverlap: overlap,
      });

      setMessage(
        `Uploaded successfully: ${document.name}`
      );

      setFile(null);

      // Reset the browser file input.
      event.target.reset();

      if (onUploadSuccess) {
        onUploadSuccess(document);
      }
    } catch (err) {
      console.error("Document upload failed:", err);

      setError(
        err.response?.data?.detail ||
          "Document upload failed. Please try again."
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="document-upload">
      <div className="document-upload-header">
        <div>
          <h3>Upload Knowledge Document</h3>
          <p>
            Add a PDF to your tenant knowledge base.
          </p>
        </div>
      </div>

      <form onSubmit={handleUpload}>
        <div className="document-upload-field">
          <label htmlFor="document-file">
            PDF Document
          </label>

          <input
            id="document-file"
            type="file"
            accept="application/pdf,.pdf"
            onChange={handleFileChange}
            disabled={uploading}
          />

          {file && (
            <span className="document-selected-file">
              Selected: {file.name}
            </span>
          )}
        </div>

        <div className="document-upload-grid">
          <div className="document-upload-field">
            <label htmlFor="chunk-size">
              Chunk Size
            </label>

            <input
              id="chunk-size"
              type="number"
              min="300"
              max="2000"
              value={chunkSize}
              onChange={(event) =>
                setChunkSize(event.target.value)
              }
              disabled={uploading}
            />

            <small>
              Range: 300–2000
            </small>
          </div>

          <div className="document-upload-field">
            <label htmlFor="chunk-overlap">
              Chunk Overlap
            </label>

            <input
              id="chunk-overlap"
              type="number"
              min="0"
              max="500"
              value={chunkOverlap}
              onChange={(event) =>
                setChunkOverlap(event.target.value)
              }
              disabled={uploading}
            />

            <small>
              Range: 0–500
            </small>
          </div>
        </div>

        <button
          type="submit"
          className="document-upload-button"
          disabled={uploading || !file}
        >
          {uploading ? "Uploading & Indexing..." : "Upload PDF"}
        </button>
      </form>

      {message && (
        <div className="document-upload-success">
          {message}
        </div>
      )}

      {error && (
        <div className="document-upload-error">
          {error}
        </div>
      )}
    </section>
  );
}