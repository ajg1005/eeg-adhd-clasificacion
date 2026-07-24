import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type Dispatch,
  type SetStateAction,
} from "react";

import type { TabId } from "../../app/tabs";
import {
  getHealth,
  getModelFigures,
  getModelInfo,
  getModels,
  predictCsv,
  validateCsv,
} from "../../api";
import type {
  ApiStatus,
  CvMetrics,
  MetricChartDatum,
  ModelFigure,
  ModelInfo,
  ModelMetrics,
  ModelRegistryItem,
  PredictionResult,
  ValidationResult,
} from "../../types";

const DEFAULT_MODEL_ID = "ml_best";

interface UseInferenceControllerResult {
  activeTab: TabId;
  apiStatus: ApiStatus;
  decisionScore: number | null;
  error: string;
  file: File | null;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  handleModelChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  handlePrediction: () => Promise<void>;
  loadingPrediction: boolean;
  loadingValidation: boolean;
  metrics: CvMetrics | ModelMetrics | null;
  metricsChartData: MetricChartDatum[];
  modelFigures: ModelFigure[];
  modelInfo: ModelInfo | null;
  models: ModelRegistryItem[];
  prediction: PredictionResult | null;
  refreshModels: (
    preferredModelId?: string | null,
  ) => Promise<ModelRegistryItem[]>;
  selectedModelId: string;
  setActiveTab: Dispatch<SetStateAction<TabId>>;
  validation: ValidationResult | null;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function isModelEnabled(model: ModelRegistryItem): boolean {
  return model.enabled !== false;
}

function chooseModelId(
  availableModels: ModelRegistryItem[],
  preferredModelId: string | null,
  currentModelId: string,
): string {
  const enabledModels = availableModels.filter(isModelEnabled);
  const candidates = [preferredModelId, currentModelId, DEFAULT_MODEL_ID];
  const selectedCandidate = candidates.find(
    (candidate) =>
      candidate !== null &&
      enabledModels.some((model) => model.model_id === candidate),
  );

  return selectedCandidate ?? enabledModels[0]?.model_id ?? "";
}

// Controlador del flujo de inferencia: selección de modelo, validación del CSV
// del paciente y predicción.
export function useInferenceController(): UseInferenceControllerResult {
  const [activeTab, setActiveTab] = useState<TabId>("dataset");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState("");
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [modelFigures, setModelFigures] = useState<ModelFigure[]>([]);

  const refreshModels = useCallback(
    async (
      preferredModelId: string | null = null,
    ): Promise<ModelRegistryItem[]> => {
      const availableModels = await getModels();
      setModels(availableModels);
      setSelectedModelId((currentModelId) =>
        chooseModelId(availableModels, preferredModelId, currentModelId),
      );

      return availableModels;
    },
    [],
  );

  // Datos estáticos del backend: se cargan al montar el hook.
  useEffect(() => {
    let cancelled = false;

    void Promise.all([getHealth(), getModels()])
      .then(([, availableModels]) => {
        if (cancelled) {
          return;
        }

        setApiStatus("ok");
        setModels(availableModels);
        setSelectedModelId((currentModelId) =>
          chooseModelId(availableModels, null, currentModelId),
        );
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setApiStatus("error");
          setError(
            errorMessage(caughtError, "No se pudo conectar con la API"),
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Info y figuras del modelo: se recargan cada vez que cambia el seleccionado.
  useEffect(() => {
    if (!selectedModelId) {
      return;
    }

    let cancelled = false;

    void Promise.all([
      getModelInfo(selectedModelId),
      getModelFigures(selectedModelId),
    ])
      .then(([info, figures]) => {
        if (!cancelled) {
          setModelInfo(info);
          setModelFigures(figures);
        }
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(
              caughtError,
              "No se pudo cargar la información del modelo",
            ),
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  async function revalidateFile(
    modelId: string,
    fileToValidate: File | null,
  ): Promise<void> {
    if (!modelId || !fileToValidate) {
      return;
    }

    setLoadingValidation(true);

    try {
      const result = await validateCsv(fileToValidate, modelId);
      setValidation(result);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "No se pudo validar el CSV"));
    } finally {
      setLoadingValidation(false);
    }
  }

  function handleModelChange(event: ChangeEvent<HTMLSelectElement>): void {
    const nextModelId = event.target.value;
    setSelectedModelId(nextModelId);
    setModelInfo(null);
    setPrediction(null);
    setValidation(null);
    setModelFigures([]);
    setError("");

    void revalidateFile(nextModelId, file);
  }

  async function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const selectedFile = event.target.files?.[0] ?? null;

    setFile(selectedFile);
    setValidation(null);
    setPrediction(null);
    setError("");

    await revalidateFile(selectedModelId, selectedFile);
  }

  async function handlePrediction(): Promise<void> {
    if (!selectedModelId) {
      setError("No hay ningún modelo disponible para realizar la predicción.");
      return;
    }

    if (!file) {
      setError("Primero sube un archivo CSV.");
      return;
    }

    setLoadingPrediction(true);
    setError("");

    try {
      const result = await predictCsv(file, selectedModelId);
      setPrediction(result);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "No se pudo realizar la predicción"));
    } finally {
      setLoadingPrediction(false);
    }
  }

  const activeModelInfo = selectedModelId ? modelInfo : null;
  const metrics =
    activeModelInfo?.metrics?.cv_metrics ?? activeModelInfo?.metrics ?? null;

  const decisionScore = prediction
    ? (prediction.decision_score ?? prediction.confidence ?? null)
    : null;

  const metricsChartData = useMemo<MetricChartDatum[]>(() => {
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
      value: Number((item.value ?? 0).toFixed(3)),
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
    modelFigures: selectedModelId ? modelFigures : [],
    modelInfo: activeModelInfo,
    models,
    prediction,
    refreshModels,
    selectedModelId,
    setActiveTab,
    validation,
  };
}
