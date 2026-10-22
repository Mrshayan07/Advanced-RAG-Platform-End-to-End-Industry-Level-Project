
import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

import {
  getCurrentUser,
  getStoredToken,
  getStoredUser,
  loginUser,
  logoutUser,
  saveAuth,
} from "../services/auth";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(getStoredUser);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initializeAuth = async () => {
      const token = getStoredToken();

      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const currentUser = await getCurrentUser();

        setUser(currentUser);
        localStorage.setItem(
          "user",
          JSON.stringify(currentUser)
        );
      } catch {
        logoutUser();
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (email, password) => {
    const data = await loginUser(email, password);

    /*
     * FastAPI OAuth/JWT APIs may return the token using
     * access_token. We keep the frontend flexible enough
     * to handle a nested user object if the backend provides it.
     */
    const token = data.access_token;

    if (!token) {
      throw new Error(
        "Login successful response did not contain access_token."
      );
    }

    saveAuth(token, data.user || null);

    let currentUser = data.user || null;

    if (!currentUser) {
      try {
        currentUser = await getCurrentUser();

        localStorage.setItem(
          "user",
          JSON.stringify(currentUser)
        );
      } catch {
        // Token is still stored; /auth/me can be retried later.
      }
    }

    setUser(currentUser);

    return {
      ...data,
      user: currentUser,
    };
  };

  const logout = () => {
    logoutUser();
    setUser(null);
  };

  const value = {
    user,
    loading,
    isAuthenticated: Boolean(getStoredToken()),
    login,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error(
      "useAuth must be used inside AuthProvider."
    );
  }

  return context;
}
