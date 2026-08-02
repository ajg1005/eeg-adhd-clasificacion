import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { PropsWithChildren } from "react";

import {
  getCurrentUser,
  login as requestLogin,
  register as requestRegistration,
} from "./api";
import { AuthContext } from "./auth-context";
import type { AuthContextValue } from "./auth-context";
import {
  AUTH_SESSION_EXPIRED_EVENT,
  clearAccessToken,
  getAccessToken,
  saveAccessToken,
} from "./session";
import type {
  AuthUser,
  LoginCredentials,
  RegisterCredentials,
} from "./types";

export function AuthProvider({ children }: PropsWithChildren) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(
    () => getAccessToken() !== null,
  );

  useEffect(() => {
    let active = true;

    function handleExpiredSession(): void {
      setUser(null);
      setIsLoading(false);
    }

    window.addEventListener(
      AUTH_SESSION_EXPIRED_EVENT,
      handleExpiredSession,
    );

    if (getAccessToken()) {
      void getCurrentUser()
        .then((currentUser) => {
          if (active) {
            setUser(currentUser);
          }
        })
        .catch(() => {
          clearAccessToken();
          if (active) {
            setUser(null);
          }
        })
        .finally(() => {
          if (active) {
            setIsLoading(false);
          }
        });
    }

    return () => {
      active = false;
      window.removeEventListener(
        AUTH_SESSION_EXPIRED_EVENT,
        handleExpiredSession,
      );
    };
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    const token = await requestLogin(credentials);
    saveAccessToken(token.access_token);

    try {
      const currentUser = await getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      clearAccessToken();
      throw error;
    }
  }, []);

  const register = useCallback(
    async (credentials: RegisterCredentials) => {
      await requestRegistration(credentials);
      await login(credentials);
    },
    [login],
  );

  const logout = useCallback(() => {
    clearAccessToken();
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      isLoading,
      login,
      logout,
      register,
      user,
    }),
    [isLoading, login, logout, register, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}