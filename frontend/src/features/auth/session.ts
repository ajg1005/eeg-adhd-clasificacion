const ACCESS_TOKEN_STORAGE_KEY = "eeg-adhd.access-token";

export const AUTH_SESSION_EXPIRED_EVENT = "eeg-adhd:auth-session-expired";

export function getAccessToken(): string | null {
  try {
    return window.sessionStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveAccessToken(token: string): void {
  window.sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
}

export function clearAccessToken(): void {
  window.sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
}

export function expireSession(): void {
  const hadActiveSession = getAccessToken() !== null;
  clearAccessToken();

  if (hadActiveSession) {
    window.dispatchEvent(new Event(AUTH_SESSION_EXPIRED_EVENT));
  }
}