import { requestJson } from "../../shared/api/client";
import { translate } from "../../shared/utils/errors";
import type {
  AuthToken,
  AuthUser,
  LoginCredentials,
  RegisterCredentials,
} from "./types";

export function login(credentials: LoginCredentials): Promise<AuthToken> {
  const formData = new URLSearchParams({
    username: credentials.email,
    password: credentials.password,
  });

  return requestJson<AuthToken>(
    { route: "authLogin" },
    {
      method: "POST",
      body: formData,
    },
    translate("errors.auth.login"),
  );
}

export function register(
  credentials: RegisterCredentials,
): Promise<AuthUser> {
  return requestJson<AuthUser>(
    { route: "authRegister" },
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(credentials),
    },
    translate("errors.auth.register"),
  );
}

export function getCurrentUser(): Promise<AuthUser> {
  return requestJson<AuthUser>(
    { route: "authMe" },
    undefined,
    translate("errors.auth.profile"),
  );
}