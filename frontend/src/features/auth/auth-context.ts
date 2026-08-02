import { createContext } from "react";

import type {
  AuthUser,
  LoginCredentials,
  RegisterCredentials,
} from "./types";

export interface AuthContextValue {
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  register: (credentials: RegisterCredentials) => Promise<void>;
  user: AuthUser | null;
}

export const AuthContext = createContext<AuthContextValue | null>(null);