const SIGNAL_PARAM_LABELS = {
  epoch_size: "Tamaño de ventana",
  step_size: "Desplazamiento entre ventanas",
  sfreq: "Frecuencia de muestreo",
  nperseg: "Muestras por segmento",
  feature_mode: "Tipo de características",
  use_filtering: "Aplicar filtrado",
};

const MODEL_PARAM_LABELS = {
  C: "Parámetro C",
  gamma: "Gamma",
  n_estimators: "Número de estimadores",
  max_depth: "Profundidad máxima",
  criterion: "Criterio de división",
  class_weight: "Pesos de clase",
  learning_rate: "Tasa de aprendizaje",
  subsample: "Submuestreo de filas",
  colsample_bytree: "Submuestreo de columnas",
  filters: "Número de filtros",
  dropout: "Dropout",
  lstm_units: "Unidades LSTM",
  n_neighbors: "Número de vecinos",
  weights: "Ponderación",
  max_iter: "Iteraciones máximas",
};

const TRAINING_PARAM_LABELS = {
  epochs: "Épocas",
  batch_size: "Tamaño de lote",
  learning_rate: "Tasa de aprendizaje",
  early_stopping_patience: "Paciencia de parada temprana",
};

const VALUE_LABELS = {
  auto: "Automático",
  balanced: "Balanceado",
  combined: "Temporales y espectrales",
  distance: "Por distancia",
  entropy: "Entropía",
  false: "No",
  gini: "Gini",
  scale: "Escala",
  spectral: "Espectrales",
  sqrt: "Raíz cuadrada",
  temporal: "Temporales",
  true: "Sí",
  uniform: "Uniforme",
};

const EVALUATION_MODE_LABELS = {
  cross_subject_cv: "Validación cruzada por paciente",
  stratified_group_k_fold: "Validación cruzada estratificada por paciente",
};

const METHOD_LABELS = {
  permutation_importance: "Importancia por permutación",
  f1_weighted: "F1 ponderado",
  f1_macro: "F1 macro",
  accuracy: "Exactitud",
  balanced_accuracy: "Exactitud balanceada",
};

const REPORT_ROW_LABELS = {
  "0": "Control",
  "1": "TDAH",
  Control: "Control",
  ADHD: "TDAH",
  accuracy: "Exactitud",
  "macro avg": "Promedio macro",
  "weighted avg": "Promedio ponderado",
};

const REPORT_COLUMN_LABELS = {
  precision: "Precisión",
  recall: "Sensibilidad",
  "f1-score": "F1-score",
  support: "Muestras",
};

export function signalParamLabel(name) {
  return SIGNAL_PARAM_LABELS[name] || name;
}

export function modelParamLabel(name) {
  return MODEL_PARAM_LABELS[name] || name;
}

export function trainingParamLabel(name) {
  return TRAINING_PARAM_LABELS[name] || name;
}

export function optionValueLabel(value) {
  if (value === null || value === undefined || value === "none") {
    return "Sin valor";
  }

  return VALUE_LABELS[String(value)] || String(value);
}

export function evaluationModeLabel(value) {
  return EVALUATION_MODE_LABELS[value] || value;
}

export function methodLabel(value) {
  return METHOD_LABELS[value] || value;
}

export function reportRowLabel(name) {
  return REPORT_ROW_LABELS[name] || name;
}

export function reportColumnLabel(name) {
  return REPORT_COLUMN_LABELS[name] || name;
}

// Columnas estandar del classification_report de scikit-learn.
export const REPORT_COLUMNS = ["precision", "recall", "f1-score", "support"];
