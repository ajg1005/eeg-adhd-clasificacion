import type { AsyncTaskResponse } from "../../shared/types";
import { requestJson } from "../../shared/api/client";
import type {
  TrainingOptions,
  TrainingPayload,
} from "./types";

export function getTrainingOptions(): Promise<TrainingOptions> {
  return requestJson<TrainingOptions>(
    "/training/options",
    undefined,
    "No se pudieron cargar los parámetros de entrenamiento",
  );
}

export function runTraining(
  file: File | null | undefined,
  payload: TrainingPayload,
): Promise<AsyncTaskResponse> {
  const formData = new FormData();

  if (payload.datasetId) {
    formData.append("dataset_id", String(payload.datasetId));
  } else if (file) {
    formData.append("file", file);
  }

  formData.append("model_type", payload.modelType);
  formData.append("model_name", payload.modelName);
  formData.append("eeg_params", JSON.stringify(payload.eegParams));
  formData.append("model_params", JSON.stringify(payload.modelParams));
  formData.append("training_params", JSON.stringify(payload.trainingParams));

  return requestJson<AsyncTaskResponse>(
    "/training/run",
    { method: "POST", body: formData },
    "No se pudo entrenar el modelo",
  );
}
