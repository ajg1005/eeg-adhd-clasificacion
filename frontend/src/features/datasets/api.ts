import type { AsyncTaskResponse } from "../../shared/types";
import {
  positiveIntegerPathSegment,
  requestJson,
} from "../../shared/api/client";
import type { SavedTrainingDataset } from "./types";

export async function getSavedTrainingDatasets(): Promise<
  SavedTrainingDataset[]
> {
  const data = await requestJson<{ datasets: SavedTrainingDataset[] }>(
    "/training/datasets",
    undefined,
    "No se pudieron cargar los datasets guardados",
  );
  return data.datasets;
}

export function uploadTrainingDataset(
  file: File,
): Promise<SavedTrainingDataset> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<SavedTrainingDataset>(
    "/training/datasets",
    { method: "POST", body: formData },
    "No se pudo guardar el dataset",
  );
}

export function startDatasetAnalysis(
  datasetId: number,
): Promise<AsyncTaskResponse> {
  const safeDatasetId = positiveIntegerPathSegment(
    datasetId,
    "Identificador de dataset no válido",
  );

  return requestJson<AsyncTaskResponse>(
    `/training/datasets/${safeDatasetId}/analysis`,
    { method: "POST" },
    "No se pudo iniciar el análisis",
  );
}
