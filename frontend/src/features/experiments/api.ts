import {
  positiveIntegerPathSegment,
  requestJson,
} from "../../shared/api/client";
import type {
  BestAvailableModel,
  ExperimentDetail,
  ExperimentSummary,
} from "./types";

export function getBestAvailableModel(): Promise<BestAvailableModel> {
  return requestJson<BestAvailableModel>(
    "/models/best",
    undefined,
    "No se pudo cargar el mejor modelo disponible",
  );
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  const data = await requestJson<{ experiments: ExperimentSummary[] }>(
    "/experiments",
    undefined,
    "No se pudo cargar el historial de experimentos",
  );
  return data.experiments;
}

export function getExperimentDetail(
  experimentId: number,
): Promise<ExperimentDetail> {
  const safeExperimentId = positiveIntegerPathSegment(
    experimentId,
    "Identificador de experimento no válido",
  );

  return requestJson<ExperimentDetail>(
    `/experiments/${safeExperimentId}`,
    undefined,
    "No se pudo cargar el experimento",
  );
}
