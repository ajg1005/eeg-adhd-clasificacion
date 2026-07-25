import { requestJson, resolveApiAsset } from "../../shared/api/client";
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
    "No se pudieron cargar los modelos disponibles",
  );
  return data.models;
}

export function getModelInfo(modelId = "ml_best"): Promise<ModelInfo> {
  return requestJson<ModelInfo>(
    { route: "modelInfo", query: { model_id: modelId } },
    undefined,
    "No se pudo cargar la información del modelo",
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
    "CSV no válido",
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
    "Error durante la predicción",
  );
}

export async function getModelFigures(
  modelId = "ml_best",
): Promise<ModelFigure[]> {
  const data = await requestJson<{ figures: ModelFigure[] }>(
    { route: "modelFigures", query: { model_id: modelId } },
    undefined,
    "No se pudieron cargar las figuras del modelo",
  );

  return data.figures.flatMap((figure) => {
    const url = resolveApiAsset(figure.url);

    return url ? [{ ...figure, url }] : [];
  });
}
