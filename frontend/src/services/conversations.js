import api from "./api";

export async function getConversations() {
  const response = await api.get("/conversations");

  return response.data;
}

export async function getConversation(conversationId) {
  const response = await api.get(
    `/conversations/${conversationId}`
  );

  return response.data;
}

export async function deleteConversation(
  conversationId
) {
  await api.delete(
    `/conversations/${conversationId}`
  );
}