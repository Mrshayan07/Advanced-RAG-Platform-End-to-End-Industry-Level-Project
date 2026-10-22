import api from "./api";

export async function queryRAG({
  question,
  conversationId = null,
  chatHistory = "",
}) {
  const response = await api.post("/rag/query", {
    question,
    conversation_id: conversationId,
    chat_history: chatHistory,
  });

  return response.data;
}