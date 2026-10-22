import { useState } from "react";

export default function ChatInput({
  onSend,
  disabled = false,
}) {
  const [question, setQuestion] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || disabled) {
      return;
    }

    setQuestion("");

    await onSend(trimmedQuestion);
  };

  const handleKeyDown = (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form
      className="chat-input-form"
      onSubmit={handleSubmit}
    >
      <textarea
        value={question}
        onChange={(event) =>
          setQuestion(event.target.value)
        }
        onKeyDown={handleKeyDown}
        placeholder="Ask something about your documents..."
        rows={1}
        disabled={disabled}
      />

      <button
        type="submit"
        className="send-button"
        disabled={
          disabled ||
          !question.trim()
        }
      >
        {disabled ? "..." : "Send"}
      </button>
    </form>
  );
}