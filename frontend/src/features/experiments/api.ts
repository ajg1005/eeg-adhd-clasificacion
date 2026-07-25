import { requestJson } from "../../shared/api/client";
import type {
  BestAvailableModel,
  ExperimentDetail,
  ExperimentSummary,
} from "./types";

export function getBestAvailableModel(): Promise<BestAvailableModel> {
  return requestJson<BestAvailableModel>(
    { route: "bestModel" },
    undefined,
    "No se pudo cargar el mejor modelo disponible",
  );
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  const data = await requestJson<{ experiments: ExperimentSummary[] }>(
    { route: "experiments" },
    undefined,
    "No se pudo cargar el historial de experimentos",
  );
  return data.experiments;
}

export function getExperimentDetail(
  experimentId: number,
): Promise<ExperimentDetail> {
  return requestJson<ExperimentDetail>(
    { route: "experimentDetail", id: experimentId },
    undefined,
    "No se pudo cargar el experimento",
  );
}
