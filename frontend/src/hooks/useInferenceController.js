import { useEffect, useMemo, useState } from "react";

import {
  getHealth,
  getModelCatalog,
  getModelFigures,
  getModelInfo,
  getModels,
  predictCsv,
  validateCsv,
} from "../api";

// Controlador del flujo de inferencia: seleccion de modelo, validacion del CSV
// del paciente y prediccion.
export function useInferenceController() {
  const [activeTab, setActiveTab] = useState("Modelo");
  const [apiStatus, setApiStatus] = useState("checking");
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("ml_best");
  const [modelInfo, setModelInfo] = useState(null);
  const [modelCatalog, setModelCatalog] = useState(null);
  const [file, setFile] = useState(null);
  const [validation, setValidation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [modelFigures, setModelFigures] = useState([]);

  useEffect(() => {
    async function loadInitialData() {
      try {
        await getHealth();
        setApiStatus("ok");

        const availableModels = await getModels();
        setModels(availableModels);

        const catalog = await getModelCatalog();
        setModelCatalog(catalog);

        const info = await getModelInfo(selectedModelId);
        setModelInfo(info);

        const figures = await getModelFigures(selectedModelId);
        setModelFigures(figures);
      } catch (err) {
        setApiStatus("error");
        setError(err.message);
      }
    }

    loadInitialData();
  }, [selectedModelId]);

  async function revalidateFile(modelId, fileToValidate) {
    if (!fileToValidate) {
      return;
    }
    setLoadingValidation(true);
    try {
      const result = await validateCsv(fileToValidate, modelId);
      setValidation(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingValidation(false);
    }
  }

  function handleModelChange(event) {
    const nextModelId = event.target.value;
    setSelectedModelId(nextModelId);
    setModelInfo(null);
    setPrediction(null);
    setValidation(null);
    setModelFigures([]);
    setError("");
    // Si ya hay archivo subido, lo re-validamos contra el modelo nuevo
    revalidateFile(nextModelId, file);
  }

  async function handleFileChange(event) {
    const selectedFile = event.target.files[0] || null;

    setFile(selectedFile);
    setValidation(null);
    setPrediction(null);
    setError("");

    await revalidateFile(selectedModelId, selectedFile);
  }

  async function handlePrediction() {
    if (!file) {
      setError("Primero sube un archivo CSV.");
      return;
    }

    setLoadingPrediction(true);
    setError("");

    try {
      const result = await predictCsv(file, selectedModelId);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPrediction(false);
    }
  }

  const metrics = modelInfo?.metrics?.cv_metrics || modelInfo?.metrics;

  const predictionChartData = useMemo(() => {
    if (!prediction) {
      return [];
    }

    return Object.entries(prediction.epoch_percentage_by_class).map(
      ([label, percentage]) => ({
        clase: label,
        porcentaje: Number((percentage * 100).toFixed(2)),
        epochs: prediction.epoch_count_by_class[label] || 0,
      })
    );
  }, [prediction]);

  const decisionScore = prediction
    ? prediction.decision_score ?? prediction.confidence
    : null;

  const metricsChartData = useMemo(() => {
    if (!metrics) {
      return [];
    }

    return [
      { name: "Accuracy", value: metrics.accuracy_epoch_mean },
      { name: "Balanced", value: metrics.balanced_accuracy_epoch_mean },
      { name: "Precision", value: metrics.precision_epoch_mean },
      { name: "Recall", value: metrics.recall_epoch_mean },
      { name: "F1", value: metrics.f1_epoch_mean },
    ].map((item) => ({
      ...item,
      value: Number((item.value || 0).toFixed(3)),
    }));
  }, [metrics]);

  return {
    activeTab,
    apiStatus,
    decisionScore,
    error,
    file,
    handleFileChange,
    handleModelChange,
    handlePrediction,
    loadingPrediction,
    loadingValidation,
    metrics,
    metricsChartData,
    modelCatalog,
    modelFigures,
    modelInfo,
    models,
    prediction,
    predictionChartData,
    selectedModelId,
    setActiveTab,
    validation,
  };
}
