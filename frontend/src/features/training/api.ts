import type { AsyncTaskResponse } from "../../shared/types";
import { requestJson } from "../../shared/api/client";
import { translate } from "../../shared/utils/errors";
import type {
  TrainingOptions,
  TrainingPayload,
} from "./types";

export function getTrainingOptions(): Promise<TrainingOptions> {
  return requestJson<TrainingOptions>(
    { route: "trainingOptions" },
    undefined,
    translate("errors.training.params"),
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
    { route: "trainingRun" },
    { method: "POST", body: formData },
    translate("errors.training.run"),
  );
}
