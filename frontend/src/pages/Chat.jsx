import {
  useEffect,
  useRef,
  useState,
} from "react";

import { useAuth } from "../context/AuthContext";
import { queryRAG } from "../services/rag";
import { getConversation } from "../services/conversations";

import ChatInput from "../components/ChatInput";
import ChatMessage from "../components/ChatMessage";
import ConversationSidebar from "../components/ConversationSidebar";

export default function Chat() {
  const { user, logout } = useAuth();

  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] =
    useState(null);

  const [loading, setLoading] = useState(false);
  const [loadingConversation, setLoadingConversation] =
    useState(false);

  const [error, setError] = useState("");

  const [
    conversationRefreshTrigger,
    setConversationRefreshTrigger,
  ] = useState(0);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  const handleSend = async (question) => {
    setError("");

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: question,
        citations: [],
      },
    ]);

    setLoading(true);

    try {
      const response = await queryRAG({
        question,
        conversationId,
      });

      if (response.conversation_id) {
        setConversationId(
          response.conversation_id
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: response.answer,
          citations:
            response.citations || [],
        },
      ]);

      /*
       * Refresh sidebar after the backend has
       * persisted the conversation turn.
       */
      setConversationRefreshTrigger(
        (previous) => previous + 1
      );
    } catch (err) {
      console.error(
        "RAG query failed:",
        err
      );

      const detail =
        err.response?.data?.detail ||
        err.message ||
        "Failed to get a response from the RAG system.";

      setError(
        typeof detail === "string"
          ? detail
          : "Failed to get a response from the RAG system."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setConversationId(null);
    setError("");
  };

  const handleSelectConversation = async (
    selectedConversationId
  ) => {
    if (
      selectedConversationId ===
      conversationId
    ) {
      return;
    }

    setError("");
    setLoadingConversation(true);

    try {
      const conversation =
        await getConversation(
          selectedConversationId
        );

      const restoredMessages =
        (conversation.messages || []).map(
          (message) => ({
            role: message.role,
            content: message.content,
            citations:
              message.citations || [],
          })
        );

      setConversationId(
        conversation.id
      );

      setMessages(restoredMessages);
    } catch (err) {
      console.error(
        "Failed to load conversation:",
        err
      );

      const detail =
        err.response?.data?.detail ||
        err.message ||
        "Failed to load conversation.";

      setError(
        typeof detail === "string"
          ? detail
          : "Failed to load conversation."
      );
    } finally {
      setLoadingConversation(false);
    }
  };

  return (
    <main className="chat-page">
      <ConversationSidebar
        activeConversationId={
          conversationId
        }
        onSelectConversation={
          handleSelectConversation
        }
        onNewChat={handleNewChat}
        refreshTrigger={
          conversationRefreshTrigger
        }
      />

      <section className="chat-main">
        <header className="chat-header">
          <div className="chat-brand">
            <div className="brand-mark">
              R
            </div>

            <div>
              <h1>Advanced RAG</h1>
              <p>
                AI Knowledge Assistant
              </p>
            </div>
          </div>

          <div className="chat-actions">
            {user?.role && (
              <span className="role-badge">
                {user.role}
              </span>
            )}

            <button
              type="button"
              className="secondary-button"
              onClick={handleNewChat}
            >
              New Chat
            </button>

            <button
              type="button"
              className="secondary-button"
              onClick={logout}
            >
              Logout
            </button>
          </div>
        </header>

        <section className="chat-container">
          <div className="messages-container">
            {loadingConversation ? (
              <div className="chat-loading-state">
                Loading conversation...
              </div>
            ) : messages.length === 0 &&
              !loading ? (
              <div className="chat-empty-state">
                <div className="chat-empty-icon">
                  AI
                </div>

                <h2>
                  How can I help you?
                </h2>

                <p>
                  Ask questions about your
                  knowledge base.
                </p>

                <div className="suggestion-list">
                  <button
                    type="button"
                    onClick={() =>
                      handleSend(
                        "What information is available in the knowledge base?"
                      )
                    }
                  >
                    What information is
                    available?
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      handleSend(
                        "Summarize the available documents."
                      )
                    }
                  >
                    Summarize the documents
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      handleSend(
                        "What technologies are mentioned in the documents?"
                      )
                    }
                  >
                    What technologies are
                    mentioned?
                  </button>
                </div>
              </div>
            ) : (
              <>
                {messages.map(
                  (message, index) => (
                    <ChatMessage
                      key={`${conversationId}-${index}`}
                      role={message.role}
                      content={
                        message.content
                      }
                      citations={
                        message.citations
                      }
                    />
                  )
                )}

                {loading && (
                  <div className="message-row message-row-assistant">
                    <div className="message-avatar assistant-avatar">
                      AI
                    </div>

                    <div className="message-bubble assistant-message">
                      <div className="typing-indicator">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {error && (
            <div className="chat-error">
              <strong>Error:</strong>{" "}
              {error}
            </div>
          )}

          <div className="chat-input-container">
            <ChatInput
              onSend={handleSend}
              disabled={
                loading ||
                loadingConversation
              }
            />

            <p className="input-hint">
              Press Enter to send · Shift +
              Enter for a new line
            </p>
          </div>
        </section>
      </section>
    </main>
  );
}