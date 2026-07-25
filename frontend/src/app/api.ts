import { requestJson } from "../shared/api/client";
import type { HealthResponse } from "./types";
import { translate } from "../shared/utils/errors";

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(
    { route: "health" },
    undefined,
    translate("errors.health"),
  );
}
