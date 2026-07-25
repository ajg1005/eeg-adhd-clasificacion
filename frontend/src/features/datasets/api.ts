import type { AsyncTaskResponse } from "../../shared/types";
import { requestJson } from "../../shared/api/client";
import type { SavedTrainingDataset } from "./types";

export async function getSavedTrainingDatasets(): Promise<
  SavedTrainingDataset[]
> {
  const data = await requestJson<{ datasets: SavedTrainingDataset[] }>(
    { route: "trainingDatasets" },
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
    { route: "trainingDatasets" },
    { method: "POST", body: formData },
    "No se pudo guardar el dataset",
  );
}

export function startDatasetAnalysis(
  datasetId: number,
): Promise<AsyncTaskResponse> {
  return requestJson<AsyncTaskResponse>(
    { route: "datasetAnalysis", id: datasetId },
    { method: "POST" },
    "No se pudo iniciar el análisis",
  );
}
