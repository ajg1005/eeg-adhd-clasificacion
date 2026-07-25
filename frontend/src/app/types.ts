export type ApiStatus = "checking" | "ok" | "error";

export interface HealthResponse {
  status: string;
}
