import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type Dispatch,
  type SetStateAction,
} from "react";

import type { TabId } from "../../app/tabs";
import {
  getModelFigures,
  getModelInfo,
  getModels,
  predictCsv,
  validateCsv,
} from "./api";
import type {
  CvMetrics,
  MetricChartDatum,
  ModelFigure,
  ModelInfo,
  ModelMetrics,
  ModelRegistryItem,
  PredictionResult,
  ValidationResult,
} from "./types";
import { errorMessage, translate } from "../../shared/utils/errors";

const DEFAULT_MODEL_ID = "ml_best";

interface UseInferenceControllerResult {
  activeTab: TabId;
  decisionScore: number | null;
  error: string;
  file: File | null;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  handleModelChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  selectModel: (modelId: string) => void;
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
  const validationRequestRef = useRef<AbortController | null>(null);
  const predictionRequestRef = useRef<AbortController | null>(null);

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

    void getModels()
      .then((availableModels) => {
        if (cancelled) {
          return;
        }

        setModels(availableModels);
        setSelectedModelId((currentModelId) =>
          chooseModelId(availableModels, null, currentModelId),
        );
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(errorMessage(caughtError, "errors.models.list"));
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

    void getModelInfo(selectedModelId)
      .then((info) => {
        if (!cancelled) {
          setModelInfo(info);
        }
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(
              caughtError,
              "errors.models.info",
            ),
          );
        }
      });

    void getModelFigures(selectedModelId)
      .then((figures) => {
        if (!cancelled) {
          setModelFigures(figures);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelFigures([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);
  useEffect(
    () => () => {
      validationRequestRef.current?.abort();
      predictionRequestRef.current?.abort();
    },
    [],
  );

  async function revalidateFile(
    modelId: string,
    fileToValidate: File | null,
  ): Promise<void> {
    validationRequestRef.current?.abort();
    validationRequestRef.current = null;

    if (!modelId || !fileToValidate) {
      setLoadingValidation(false);
      return;
    }

    setLoadingValidation(true);
    const controller = new AbortController();
    validationRequestRef.current = controller;

    try {
      const result = await validateCsv(
        fileToValidate,
        modelId,
        controller.signal,
      );

      if (!controller.signal.aborted) {
        setValidation(result);
      }
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setError(errorMessage(caughtError, "errors.prediction.validate"));
      }
    } finally {
      if (validationRequestRef.current === controller) {
        validationRequestRef.current = null;
        setLoadingValidation(false);
      }
    }
  }

  // Seleccionar por id, para poder llamarlo desde fuera del <select> (por
  // ejemplo al promocionar un experimento a inferencia).
  function selectModel(nextModelId: string): void {
    predictionRequestRef.current?.abort();
    setSelectedModelId(nextModelId);
    setModelInfo(null);
    setPrediction(null);
    setValidation(null);
    setModelFigures([]);
    setError("");

    void revalidateFile(nextModelId, file);
  }

  function handleModelChange(event: ChangeEvent<HTMLSelectElement>): void {
    selectModel(event.target.value);
  }

  async function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const selectedFile = event.target.files?.[0] ?? null;

    predictionRequestRef.current?.abort();
    setFile(selectedFile);
    setValidation(null);
    setPrediction(null);
    setError("");

    await revalidateFile(selectedModelId, selectedFile);
  }

  async function handlePrediction(): Promise<void> {
    if (!selectedModelId) {
      setError(translate("errors.prediction.noModel"));
      return;
    }

    if (!file) {
      setError(translate("errors.prediction.missingFile"));
      return;
    }

    predictionRequestRef.current?.abort();
    const controller = new AbortController();
    predictionRequestRef.current = controller;
    setLoadingPrediction(true);
    setError("");

    try {
      const result = await predictCsv(file, selectedModelId, controller.signal);

      if (!controller.signal.aborted) {
        setPrediction(result);
      }
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setError(errorMessage(caughtError, "errors.prediction.failed"));
      }
    } finally {
      if (predictionRequestRef.current === controller) {
        predictionRequestRef.current = null;
        setLoadingPrediction(false);
      }
    }
  }

  const activeModelInfo =
    modelInfo?.model_id === selectedModelId ? modelInfo : null;
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
    decisionScore,
    error,
    file,
    handleFileChange,
    handleModelChange,
    selectModel,
    handlePrediction,
    loadingPrediction,
    loadingValidation,
    metrics,
    metricsChartData,
    modelFigures: activeModelInfo ? modelFigures : [],
    modelInfo: activeModelInfo,
    models,
    prediction,
    refreshModels,
    selectedModelId,
    setActiveTab,
    validation,
  };
}
