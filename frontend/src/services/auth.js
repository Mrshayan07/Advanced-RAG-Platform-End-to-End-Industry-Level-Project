import api from "./api";

export async function registerUser(
  email,
  password,
  tenantName
) {
  const response = await api.post("/auth/register", {
    email,
    password,
    tenant_name: tenantName,
  });

  return response.data;
}

export async function loginUser(email, password) {
  const response = await api.post("/auth/login", {
    email,
    password,
  });

  return response.data;
}

export async function getCurrentUser() {
  const response = await api.get("/auth/me");

  return response.data;
}

export function saveAuth(token, user = null) {
  localStorage.setItem("access_token", token);

  if (user) {
    localStorage.setItem(
      "user",
      JSON.stringify(user)
    );
  }
}

export function getStoredToken() {
  return localStorage.getItem("access_token");
}

export function getStoredUser() {
  const user = localStorage.getItem("user");

  if (!user) {
    return null;
  }

  try {
    return JSON.parse(user);
  } catch {
    localStorage.removeItem("user");
    return null;
  }
}

export function logoutUser() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("user");
}