export default function ChatMessage({
  role,
  content,
  citations = [],
}) {
  const isUser = role === "user";

  return (
    <div
      className={`message-row ${
        isUser
          ? "message-row-user"
          : "message-row-assistant"
      }`}
    >
      <div
        className={`message-avatar ${
          isUser
            ? "user-avatar"
            : "assistant-avatar"
        }`}
      >
        {isUser ? "U" : "AI"}
      </div>

      <div
        className={`message-bubble ${
          isUser
            ? "user-message"
            : "assistant-message"
        }`}
      >
        <div className="message-content">
          {content}
        </div>

        {!isUser && citations.length > 0 && (
          <div className="citations">
            <div className="citations-title">
              Sources
            </div>

            {citations.map((citation, index) => {
              const isObject =
                typeof citation === "object" &&
                citation !== null;

              const citationNumber =
                isObject &&
                citation.citation_id !== undefined
                  ? citation.citation_id
                  : index + 1;

              const fileName =
                isObject
                  ? citation.file_name || "Unknown source"
                  : String(citation);

              const page =
                isObject
                  ? citation.page
                  : null;

              const chunkIndex =
                isObject
                  ? citation.chunk_index
                  : null;

              return (
                <div
                  key={
                    isObject &&
                    citation.citation_id !== undefined
                      ? citation.citation_id
                      : index
                  }
                  className="citation-card"
                >
                  <span className="citation-number">
                    {citationNumber}
                  </span>

                  <div className="citation-content">
                    <div
                      className="citation-file"
                      title={fileName}
                    >
                      {fileName}
                    </div>

                    {(page !== null ||
                      chunkIndex !== null) && (
                      <div className="citation-meta">
                        {page !== null && (
                          <span>
                            Page {page}
                          </span>
                        )}

                        {chunkIndex !== null && (
                          <span>
                            Chunk {chunkIndex}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}