import { useEffect, useMemo, useState } from "react";

import {
  getDatasetSummary,
  getHealth,
  getModelCatalog,
  getModelFigures,
  getModelInfo,
  getModels,
  getSignalPreview,
  predictCsv,
  validateCsv,
} from "../api";

// Controlador principal de la aplicacion: estado, eventos y llamadas a la API
export function useInferenceController() {
  const [activeTab, setActiveTab] = useState("Datos");
  const [apiStatus, setApiStatus] = useState("checking");
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("ml_best");
  const [modelInfo, setModelInfo] = useState(null);
  const [modelCatalog, setModelCatalog] = useState(null);
  const [file, setFile] = useState(null);
  const [validation, setValidation] = useState(null);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [classFilter, setClassFilter] = useState("all");
  const [maxPatients, setMaxPatients] = useState(10);
  const [prediction, setPrediction] = useState(null);
  const [signalPreview, setSignalPreview] = useState(null);
  const [selectedChannel, setSelectedChannel] = useState("Fp1");
  const [maxPoints, setMaxPoints] = useState(1000);
  const [error, setError] = useState("");
  const [loadingDatasetSummary, setLoadingDatasetSummary] = useState(false);
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [modelFigures, setModelFigures] = useState([]);

  useEffect(() => {
    async function loadInitialData() {
      try {
        // Al cargar la app se comprueba la API y se carga el modelo inicial
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

        if (info.channels?.length > 0) {
          setSelectedChannel(info.channels[0]);
        }
      } catch (err) {
        setApiStatus("error");
        setError(err.message);
      }
    }

    loadInitialData();
  }, [selectedModelId]);

  async function loadDatasetSummary(
    targetFile = file,
    targetClassFilter = classFilter,
    targetMaxPatients = maxPatients
  ) {
    // Cargar estadisticas del dataset y pacientes a mostrar
    if (!targetFile) {
      return;
    }

    setLoadingDatasetSummary(true);

    try {
      const result = await getDatasetSummary(
        targetFile,
        targetClassFilter,
        targetMaxPatients
      );
      setDatasetSummary(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingDatasetSummary(false);
    }
  }

  function handleModelChange(event) {
    // Al cambiar de modelo se limpian resultados antiguos
    setSelectedModelId(event.target.value);
    setModelInfo(null);
    setPrediction(null);
    setModelFigures([]);
    setError("");
  }

  async function handleFileChange(event) {
    // Guardar el CSV y validarlo antes de permitir la prediccion
    const selectedFile = event.target.files[0];

    setFile(selectedFile);
    setValidation(null);
    setDatasetSummary(null);
    setPrediction(null);
    setSignalPreview(null);
    setError("");

    if (!selectedFile) {
      return;
    }

    setLoadingValidation(true);

    try {
      const result = await validateCsv(selectedFile, selectedModelId);
      setValidation(result);
      await loadDatasetSummary(selectedFile, classFilter, maxPatients);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingValidation(false);
    }
  }

  function handleClassFilterChange(event) {
    const nextFilter = event.target.value;
    setClassFilter(nextFilter);
    loadDatasetSummary(file, nextFilter, maxPatients);
  }

  function handleMaxPatientsChange(event) {
    const nextMaxPatients = Number(event.target.value);
    setMaxPatients(nextMaxPatients);
    loadDatasetSummary(file, classFilter, nextMaxPatients);
  }

  async function handleLoadPreview() {
    // Cargar una muestra del canal seleccionado para la grafica
    if (!file) {
      setError("Primero sube un archivo CSV.");
      return;
    }

    setLoadingPreview(true);
    setError("");

    try {
      const result = await getSignalPreview(
        file,
        selectedChannel,
        maxPoints,
        selectedModelId
      );
      setSignalPreview(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function handlePrediction() {
    // Ejecutar inferencia con el modelo seleccionado
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

  // Datos agregados para la grafica de epochs por clase
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

  const finalClassEpochPercentage = prediction
    ? prediction.final_class_epoch_percentage ??
      prediction.epoch_percentage_by_class[prediction.prediction_label]
    : null;

  const adhdEpochs = prediction?.epoch_count_by_class?.ADHD ?? 0;
  const controlEpochs = prediction?.epoch_count_by_class?.Control ?? 0;
  const thresholdUsed =
    prediction?.threshold !== undefined && prediction?.threshold !== null
      ? Number(prediction.threshold).toFixed(3)
      : "N/A";

  // Datos de metricas para comparar el rendimiento del modelo
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
    adhdEpochs,
    apiStatus,
    classFilter,
    controlEpochs,
    datasetSummary,
    decisionScore,
    error,
    file,
    finalClassEpochPercentage,
    handleClassFilterChange,
    handleFileChange,
    handleLoadPreview,
    handleMaxPatientsChange,
    handleModelChange,
    handlePrediction,
    loadingDatasetSummary,
    loadingPrediction,
    loadingPreview,
    loadingValidation,
    maxPatients,
    maxPoints,
    metrics,
    metricsChartData,
    modelCatalog,
    modelFigures,
    modelInfo,
    models,
    prediction,
    predictionChartData,
    selectedChannel,
    selectedModelId,
    setActiveTab,
    setMaxPoints,
    setSelectedChannel,
    signalPreview,
    thresholdUsed,
    validation,
  };
}


