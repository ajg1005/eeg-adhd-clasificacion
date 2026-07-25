import { requestJson, resolveApiAsset } from "../../shared/api/client";
import { translate } from "../../shared/utils/errors";
import type {
  ModelFigure,
  ModelInfo,
  ModelRegistryItem,
  PredictionResult,
  ValidationResult,
} from "./types";

export async function getModels(): Promise<ModelRegistryItem[]> {
  const data = await requestJson<{ models: ModelRegistryItem[] }>(
    { route: "models" },
    undefined,
    translate("errors.models.list"),
  );
  return data.models;
}

export function getModelInfo(modelId = "ml_best"): Promise<ModelInfo> {
  return requestJson<ModelInfo>(
    { route: "modelInfo", query: { model_id: modelId } },
    undefined,
    translate("errors.models.info"),
  );
}

export function validateCsv(
  file: File,
  modelId = "ml_best",
): Promise<ValidationResult> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<ValidationResult>(
    { route: "validate", query: { model_id: modelId } },
    { method: "POST", body: formData },
    translate("errors.prediction.invalidCsv"),
  );
}

export function predictCsv(
  file: File,
  modelId = "ml_best",
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<PredictionResult>(
    { route: "predict", query: { model_id: modelId } },
    { method: "POST", body: formData },
    translate("errors.prediction.runFailed"),
  );
}

export async function getModelFigures(
  modelId = "ml_best",
): Promise<ModelFigure[]> {
  const data = await requestJson<{ figures: ModelFigure[] }>(
    { route: "modelFigures", query: { model_id: modelId } },
    undefined,
    translate("errors.models.figures"),
  );

  return data.figures.flatMap((figure) => {
    const url = resolveApiAsset(figure.url);

    return url ? [{ ...figure, url }] : [];
  });
}
