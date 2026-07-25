import { apiUrl, requestJson } from "../../shared/api/client";
import type {
  ModelFigure,
  ModelInfo,
  ModelRegistryItem,
  PredictionResult,
  ValidationResult,
} from "./types";

export async function getModels(): Promise<ModelRegistryItem[]> {
  const data = await requestJson<{ models: ModelRegistryItem[] }>(
    "/models",
    undefined,
    "No se pudieron cargar los modelos disponibles",
  );
  return data.models;
}

export function getModelInfo(modelId = "ml_best"): Promise<ModelInfo> {
  const params = new URLSearchParams({ model_id: modelId });
  return requestJson<ModelInfo>(
    `/model/info?${params}`,
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
  const params = new URLSearchParams({ model_id: modelId });

  return requestJson<ValidationResult>(
    `/validate?${params}`,
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
  const params = new URLSearchParams({ model_id: modelId });

  return requestJson<PredictionResult>(
    `/predict?${params}`,
    { method: "POST", body: formData },
    "Error durante la predicción",
  );
}

export async function getModelFigures(
  modelId = "ml_best",
): Promise<ModelFigure[]> {
  const params = new URLSearchParams({ model_id: modelId });
  const data = await requestJson<{ figures: ModelFigure[] }>(
    `/model/figures?${params}`,
    undefined,
    "No se pudieron cargar las figuras del modelo",
  );

  return data.figures.map((figure) => ({
    ...figure,
    url: apiUrl(figure.url),
  }));
}
