const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const UUID_PATH_SEGMENT_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function uuidPathSegment(value: string): string {
  const normalizedValue = value.trim().toLowerCase();

  if (!UUID_PATH_SEGMENT_PATTERN.test(normalizedValue)) {
    throw new Error("Identificador de tarea no válido");
  }

  return encodeURIComponent(normalizedValue);
}

export function positiveIntegerPathSegment(
  value: number,
  errorMessage: string,
): string {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(errorMessage);
  }

  return String(value);
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function readError(
  response: Response,
  fallbackMessage: string,
): Promise<string> {
  try {
    const error: unknown = await response.json();

    if (
      typeof error === "object" &&
      error !== null &&
      "detail" in error &&
      typeof error.detail === "string"
    ) {
      return error.detail;
    }
  } catch {
    // La respuesta no contiene un cuerpo JSON utilizable.
  }

  return fallbackMessage;
}

export async function requestJson<T>(
  path: string,
  options: RequestInit | undefined,
  fallbackMessage: string,
): Promise<T> {
  const response = await fetch(apiUrl(path), options);

  if (!response.ok) {
    throw new Error(await readError(response, fallbackMessage));
  }

  return response.json() as Promise<T>;
}
