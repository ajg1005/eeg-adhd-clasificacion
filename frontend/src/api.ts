import type {
  AsyncTaskResponse,
  BestAvailableModel,
  ExperimentDetail,
  ExperimentSummary,
  HealthResponse,
  ModelFigure,
  ModelInfo,
  ModelRegistryItem,
  PredictionResult,
  SavedTrainingDataset,
  TaskStatusResponse,
  TrainingOptions,
  TrainingPayload,
  TrainingResult,
  ValidationResult,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJson<T>(response: Response): Promise<T> {
  return response.json() as Promise<T>;
}

async function readError(
  response: Response,
  fallbackMessage: string,
): Promise<string> {
  try {
    const error: unknown = await response.json();

    if (
      typeof error === "object" &&
      error !== null &&
      "detail" in error &&
      typeof error.detail === "string"
    ) {
      return error.detail;
    }

    return fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("No se pudo conectar con la API");
  }

  return parseJson<HealthResponse>(response);
}

export async function getModels(): Promise<ModelRegistryItem[]> {
  const response = await fetch(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar los modelos disponibles");
  }

  const data = await parseJson<{ models: ModelRegistryItem[] }>(response);
  return data.models;
}

export async function getBestAvailableModel(): Promise<BestAvailableModel> {
  const response = await fetch(`${API_BASE_URL}/models/best`);

  if (!response.ok) {
    throw new Error(
      await readError(response, "No se pudo cargar el mejor modelo disponible"),
    );
  }

  return parseJson<BestAvailableModel>(response);
}

export async function getTrainingOptions(): Promise<TrainingOptions> {
  const response = await fetch(`${API_BASE_URL}/training/options`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar los parámetros de entrenamiento");
  }

  return parseJson<TrainingOptions>(response);
}

export async function getSavedTrainingDatasets(): Promise<
  SavedTrainingDataset[]
> {
  const response = await fetch(`${API_BASE_URL}/training/datasets`);

  if (!response.ok) {
    throw new Error(
      await readError(response, "No se pudieron cargar los datasets guardados"),
    );
  }

  const data = await parseJson<{ datasets: SavedTrainingDataset[] }>(response);
  return data.datasets;
}

export async function uploadTrainingDataset(
  file: File,
): Promise<SavedTrainingDataset> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/training/datasets`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo guardar el dataset"));
  }

  return parseJson<SavedTrainingDataset>(response);
}

export async function startDatasetAnalysis(
  datasetId: number,
): Promise<AsyncTaskResponse> {
  const response = await fetch(
    `${API_BASE_URL}/training/datasets/${datasetId}/analysis`,
    { method: "POST" },
  );

  if (!response.ok) {
    throw new Error(
      await readError(response, "No se pudo iniciar el análisis"),
    );
  }

  return parseJson<AsyncTaskResponse>(response);
}

export async function getTaskStatus<TResult = unknown>(
  taskId: string,
): Promise<TaskStatusResponse<TResult>> {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`);

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo consultar la tarea"));
  }

  return parseJson<TaskStatusResponse<TResult>>(response);
}

export async function runTraining(
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

  const response = await fetch(`${API_BASE_URL}/training/run`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo entrenar el modelo"));
  }

  return parseJson<AsyncTaskResponse>(response);
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${API_BASE_URL}/experiments`);

  if (!response.ok) {
    throw new Error(
      await readError(response, "No se pudo cargar el historial de experimentos"),
    );
  }

  const data = await parseJson<{ experiments: ExperimentSummary[] }>(response);
  return data.experiments;
}

export async function getExperimentDetail(
  experimentId: number,
): Promise<ExperimentDetail> {
  const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}`);

  if (!response.ok) {
    throw new Error(
      await readError(response, "No se pudo cargar el experimento"),
    );
  }

  return parseJson<ExperimentDetail>(response);
}

export async function getModelInfo(modelId = "ml_best"): Promise<ModelInfo> {
  const params = new URLSearchParams({ model_id: modelId });
  const response = await fetch(`${API_BASE_URL}/model/info?${params}`);

  if (!response.ok) {
    throw new Error("No se pudo cargar la información del modelo");
  }

  return parseJson<ModelInfo>(response);
}

export async function validateCsv(
  file: File,
  modelId = "ml_best",
): Promise<ValidationResult> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({ model_id: modelId });
  const response = await fetch(`${API_BASE_URL}/validate?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "CSV no válido"));
  }

  return parseJson<ValidationResult>(response);
}

export async function predictCsv(
  file: File,
  modelId = "ml_best",
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({ model_id: modelId });
  const response = await fetch(`${API_BASE_URL}/predict?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "Error durante la predicción"));
  }

  return parseJson<PredictionResult>(response);
}

export async function getModelFigures(
  modelId = "ml_best",
): Promise<ModelFigure[]> {
  const params = new URLSearchParams({ model_id: modelId });
  const response = await fetch(`${API_BASE_URL}/model/figures?${params}`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar las figuras del modelo");
  }

  const data = await parseJson<{ figures: ModelFigure[] }>(response);

  return data.figures.map((figure) => ({
    ...figure,
    url: `${API_BASE_URL}${figure.url}`,
  }));
}

export type { TrainingResult };