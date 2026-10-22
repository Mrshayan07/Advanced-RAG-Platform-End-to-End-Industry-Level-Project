import { useEffect, useState } from "react";
import {
  deleteConversation,
  getConversations,
} from "../services/conversations";
import DocumentUpload from "./DocumentUpload";
import DocumentList from "./DocumentList";

export default function ConversationSidebar({
  activeConversationId,
  onSelectConversation,
  onNewChat,
  refreshTrigger = 0,
}) {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [documentRefreshTrigger, setDocumentRefreshTrigger] =
    useState(0);

  const loadConversations = async () => {
    try {
      setError("");

      const data = await getConversations();
      setConversations(data);
    } catch (err) {
      console.error(
        "Failed to load conversations:",
        err
      );

      setError(
        err.response?.data?.detail ||
          "Failed to load conversations."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConversations();
  }, [refreshTrigger]);

  const handleDelete = async (
    event,
    conversationId
  ) => {
    event.stopPropagation();

    const confirmed = window.confirm(
      "Delete this conversation?"
    );

    if (!confirmed) return;

    try {
      setError("");

      await deleteConversation(conversationId);

      setConversations((previous) =>
        previous.filter(
          (conversation) =>
            conversation.id !== conversationId
        )
      );

      if (
        activeConversationId ===
        conversationId
      ) {
        onNewChat();
      }
    } catch (err) {
      console.error(
        "Failed to delete conversation:",
        err
      );

      setError(
        err.response?.data?.detail ||
          "Failed to delete conversation."
      );
    }
  };

  const handleUploadSuccess = () => {
    setError("");

    setDocumentRefreshTrigger(
      (previous) => previous + 1
    );
  };

  return (
    <aside className="conversation-sidebar">
      <div className="conversation-sidebar-header">
        <div className="sidebar-title">
          <div className="sidebar-brand-mark">
            R
          </div>

          <div>
            <h2>Conversations</h2>
            <p>Chat history</p>
          </div>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={onNewChat}
        >
          + New Chat
        </button>
      </div>

      {/* Document Upload */}
      <DocumentUpload
        onUploadSuccess={handleUploadSuccess}
      />

      {/* Knowledge Documents */}
      <DocumentList
        refreshTrigger={
          documentRefreshTrigger
        }
      />

      {/* Conversations */}
      <div className="conversation-list">
        {loading ? (
          <div className="conversation-status">
            Loading conversations...
          </div>
        ) : conversations.length === 0 ? (
          <div className="conversation-status">
            No conversations yet.
          </div>
        ) : (
          conversations.map(
            (conversation) => (
              <div
                key={conversation.id}
                className={`conversation-item ${
                  activeConversationId ===
                  conversation.id
                    ? "conversation-item-active"
                    : ""
                }`}
                onClick={() =>
                  onSelectConversation(
                    conversation.id
                  )
                }
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" ||
                    event.key === " "
                  ) {
                    event.preventDefault();

                    onSelectConversation(
                      conversation.id
                    );
                  }
                }}
              >
                <div className="conversation-item-content">
                  <span className="conversation-item-title">
                    {conversation.title ||
                      "New Conversation"}
                  </span>

                  <span className="conversation-item-date">
                    {new Date(
                      conversation.created_at
                    ).toLocaleDateString()}
                  </span>
                </div>

                <button
                  type="button"
                  className="conversation-delete-button"
                  onClick={(event) =>
                    handleDelete(
                      event,
                      conversation.id
                    )
                  }
                  aria-label={`Delete ${
                    conversation.title ||
                    "conversation"
                  }`}
                >
                  ×
                </button>
              </div>
            )
          )
        )}
      </div>

      {error && (
        <div className="conversation-sidebar-error">
          {error}
        </div>
      )}
    </aside>
  );
}