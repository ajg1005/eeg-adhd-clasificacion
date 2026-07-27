import { translate } from "../utils/errors";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const API_ORIGIN = new URL(API_BASE_URL).origin;

const UUID_PATH_SEGMENT_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const ABSOLUTE_URL_PATTERN = /^[a-z][a-z0-9+.-]*:/i;
const STATIC_ROUTES = {
  bestModel: "/models/best",
  experiments: "/experiments",
  health: "/health",
  modelFigures: "/model/figures",
  modelInfo: "/model/info",
  models: "/models",
  predict: "/predict",
  trainingDatasets: "/training/datasets",
  trainingOptions: "/training/options",
  trainingRun: "/training/run",
  validate: "/validate",
} as const;

const ID_ROUTES = {
  datasetAnalysis: {
    invalidIdKey: "errors.invalidDatasetId",
    segment: "positiveInteger",
    template: "/training/datasets/:id/analysis",
  },
  experimentDetail: {
    invalidIdKey: "errors.invalidExperimentId",
    segment: "positiveInteger",
    template: "/experiments/:id",
  },
  task: {
    invalidIdKey: "errors.invalidTaskId",
    segment: "uuid",
    template: "/tasks/:id",
  },
} as const;

type StaticApiRoute = keyof typeof STATIC_ROUTES;
export type IdApiRoute = keyof typeof ID_ROUTES;
type IdRouteDefinition = (typeof ID_ROUTES)[IdApiRoute];
type QueryParams = Record<string, string>;

interface IdApiRequest {
  route: IdApiRoute;
  id: string | number;
  query?: QueryParams;
}
export type ApiRequest =
  | { route: StaticApiRoute; id?: never; query?: QueryParams }
  | IdApiRequest;

function safePathSegment(
  id: string | number,
  { invalidIdKey, segment }: IdRouteDefinition,
): string {
  if (segment === "positiveInteger") {
    const numericId = typeof id === "number" ? id : Number(id);

    if (!Number.isSafeInteger(numericId) || numericId <= 0) {
      throw new Error(translate(invalidIdKey));
    }

    return String(numericId);
  }

  const normalizedId = String(id).trim().toLowerCase();

  if (!UUID_PATH_SEGMENT_PATTERN.test(normalizedId)) {
    throw new Error(translate(invalidIdKey));
  }

  return encodeURIComponent(normalizedId);
}

function isIdRequest(request: ApiRequest): request is IdApiRequest {
  return request.route in ID_ROUTES;
}

function resolvePath(request: ApiRequest): string {
  if (isIdRequest(request)) {
    const route = ID_ROUTES[request.route];

    return route.template.replace(":id", safePathSegment(request.id, route));
  }

  return STATIC_ROUTES[request.route];
}

function buildUrl(request: ApiRequest): string {
  const url = new URL(`${API_BASE_URL}${resolvePath(request)}`);
  if (url.origin !== API_ORIGIN) {
    throw new Error(translate("errors.forbiddenUrl"));
  }

  for (const [key, value] of Object.entries(request.query ?? {})) {
    url.searchParams.set(key, value);
  }

  return url.toString();
}
export function assertValidRouteId(
  route: IdApiRoute,
  id: string | number,
): void {
  safePathSegment(id, ID_ROUTES[route]);
}

export function resolveApiAsset(assetUrl: string): string | null {
  const candidate = ABSOLUTE_URL_PATTERN.test(assetUrl)
    ? assetUrl
    : `${API_BASE_URL}${assetUrl.startsWith("/") ? "" : "/"}${assetUrl}`;

  try {
    const resolved = new URL(candidate);

    return resolved.origin === API_ORIGIN ? resolved.toString() : null;
  } catch {
    return null;
  }
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
    return fallbackMessage;
  }

  return fallbackMessage;
}

export async function requestJson<T>(
  request: ApiRequest,
  options: RequestInit | undefined,
  fallbackMessage: string,
): Promise<T> {
  const url = buildUrl(request);
  let response: Response;

  try {
    response = await fetch(url, options);
  } catch (caughtError) {
    if (caughtError instanceof DOMException && caughtError.name === "AbortError") {
      throw caughtError;
    }

    throw new Error(fallbackMessage, { cause: caughtError });
  }

  if (!response.ok) {
    throw new Error(await readError(response, fallbackMessage));
  }

  return response.json() as Promise<T>;
}
