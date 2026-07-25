import { requestJson } from "../shared/api/client";
import type { HealthResponse } from "./types";

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(
    "/health",
    undefined,
    "No se pudo conectar con la API",
  );
}
