import type { AsyncTaskResponse } from "../../shared/types";
import { requestJson } from "../../shared/api/client";
import type { SavedTrainingDataset } from "./types";
import { translate } from "../../shared/utils/errors";

export async function getSavedTrainingDatasets(): Promise<
  SavedTrainingDataset[]
> {
  const data = await requestJson<{ datasets: SavedTrainingDataset[] }>(
    { route: "trainingDatasets" },
    undefined,
    translate("errors.datasets.list"),
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
    translate("errors.datasets.save"),
  );
}

export function startDatasetAnalysis(
  datasetId: number,
): Promise<AsyncTaskResponse> {
  return requestJson<AsyncTaskResponse>(
    { route: "datasetAnalysis", id: datasetId },
    { method: "POST" },
    translate("errors.datasets.analysisStart"),
  );
}
