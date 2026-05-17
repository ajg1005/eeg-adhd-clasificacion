// URL base del backend. Configurable via VITE_API_BASE_URL para soportar
// distintos entornos (dev local, Docker, despliegue). Fallback a localhost
// para que `npm run dev` funcione sin necesidad de fichero .env.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// Leer el mensaje de error que devuelve FastAPI
async function readError(response, fallbackMessage) {
  try {
    const error = await response.json();
    return error.detail || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

export async function getHealth() {
  // Comprobar que el backend esta levantado
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("No se pudo conectar con la API");
  }

  return response.json();
}

export async function getModels() {
  // Pedir al backend los modelos disponibles
  const response = await fetch(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar los modelos disponibles");
  }

  const data = await response.json();

  return data.models;
}

export async function getModelCatalog() {
  // Cargar modelos candidatos y parametros habituales de entrenamiento
  const response = await fetch(`${API_BASE_URL}/model/catalog`);

  if (!response.ok) {
    throw new Error("No se pudo cargar el catalogo de modelos");
  }

  return response.json();
}

export async function getTrainingOptions() {
  // Cargar parametros de red y entrenamiento
  const response = await fetch(`${API_BASE_URL}/training/options`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar los parametros de entrenamiento");
  }

  return response.json();
}



export async function getDatasetStats(file) {
  // Analizar estructura del CSV para la pantalla de entrenamiento
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/training/dataset/stats`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo analizar el dataset"));
  }

  return response.json();
}


export async function runTraining(file, payload) {
  // Entrenar un modelo interactivo con los parametros elegidos en la vista
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_type", payload.modelType);
  formData.append("model_name", payload.modelName);
  formData.append("eeg_params", JSON.stringify(payload.eegParams || {}));
  formData.append("model_params", JSON.stringify(payload.modelParams || {}));
  formData.append("training_params", JSON.stringify(payload.trainingParams || {}));


  const response = await fetch(`${API_BASE_URL}/training/run`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo entrenar el modelo"));
  }

  return response.json();
}

export async function getModelInfo(modelId = "ml_best") {
  // Cargar metadatos y metricas del modelo seleccionado
  const params = new URLSearchParams({
    model_id: modelId,
  });

  const response = await fetch(`${API_BASE_URL}/model/info?${params}`);

  if (!response.ok) {
    throw new Error("No se pudo cargar la informacion del modelo");
  }

  return response.json();
}

export async function validateCsv(file, modelId = "ml_best") {
  // Enviar el CSV para validar columnas y canales antes de predecir
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({
    model_id: modelId,
  });

  const response = await fetch(`${API_BASE_URL}/validate?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "CSV no valido"));
  }

  return response.json();
}

export async function predictCsv(file, modelId = "ml_best") {
  // Enviar el CSV al modelo elegido
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({
    model_id: modelId,
  });

  const response = await fetch(`${API_BASE_URL}/predict?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "Error durante la prediccion"));
  }

  return response.json();
}

export async function getModelFigures(modelId = "ml_best") {
  // Cargar las figuras ya generadas durante la evaluacion
  const params = new URLSearchParams({
    model_id: modelId,
  });

  const response = await fetch(`${API_BASE_URL}/model/figures?${params}`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar las figuras del modelo");
  }

  const data = await response.json();

  return data.figures.map((figure) => ({
    ...figure,
    url: `${API_BASE_URL}${figure.url}`,
  }));
}



