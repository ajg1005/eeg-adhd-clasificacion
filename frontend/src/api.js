const API_BASE_URL = "http://127.0.0.1:8000";

async function readError(response, fallbackMessage) {
  try {
    const error = await response.json();
    return error.detail || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("No se pudo conectar con la API");
  }

  return response.json();
}

export async function getModels() {
  const response = await fetch(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error("No se pudieron cargar los modelos disponibles");
  }

  const data = await response.json();

  return data.models;
}

export async function getModelInfo(modelId = "ml_best") {
  const params = new URLSearchParams({
    model_id: modelId,
  });

  const response = await fetch(`${API_BASE_URL}/model/info?${params}`);

  if (!response.ok) {
    throw new Error("No se pudo cargar la información del modelo");
  }

  return response.json();
}

export async function validateCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/validate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "CSV no valido"));
  }

  return response.json();
}

export async function getSignalPreview(file, channel, maxPoints = 1000) {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams({
    channel,
    max_points: String(maxPoints),
  });

  const response = await fetch(`${API_BASE_URL}/preview?${params}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readError(response, "No se pudo cargar la señal"));
  }

  return response.json();
}

export async function predictCsv(file, modelId = "ml_best") {
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
    throw new Error(await readError(response, "Error durante la predicción"));
  }

  return response.json();
}

export async function getModelFigures(modelId = "ml_best") {
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

