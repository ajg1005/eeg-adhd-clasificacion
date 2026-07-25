import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { Trans, useTranslation } from "react-i18next";

import { getTrainingOptions } from "./api";
import type { JsonPrimitive } from "../../shared/types";
import type { SavedTrainingDataset, TrainingDatasetStats } from "../datasets/types";
import type {
  TrainingControlValues,
  TrainingModelTypeId,
  TrainingOptions,
  TrainingPayload,
  TrainingResult,
  TrainingTaskStatus,
} from "./types";
import { TrainingActionBar } from "./components/TrainingActionBar";
import { TrainingEegParamsPanel } from "./components/TrainingEegParamsPanel";
import { TrainingModelPanel } from "./components/TrainingModelPanel";
import { TrainingResultsPanel } from "./components/TrainingResultsPanel";
import { errorMessage } from "../../shared/utils/errors";

interface TrainingViewProps {
  file: File | null;
  loadingTraining: boolean;
  onStartTraining: (
    file: File | null | undefined,
    payload: TrainingPayload,
  ) => Promise<void>;
  result: TrainingResult | null;
  selectedDataset: SavedTrainingDataset | null;
  stats: TrainingDatasetStats | null;
  taskError: string;
  taskStatus: TrainingTaskStatus;
  taskStatusAt: Date | null;
}

function normalizeValue(value: string): JsonPrimitive {
  if (value === "none") {
    return null;
  }

  if (value === "true") {
    return true;
  }

  if (value === "false") {
    return false;
  }

  const numeric = Number(value);
  return Number.isNaN(numeric) || value === "" ? value : numeric;
}

function modelDefaults(
  options: TrainingOptions,
  modelType: TrainingModelTypeId,
  modelName: string,
): TrainingControlValues {
  return options.model_types[modelType].models[modelName]?.default_params ?? {};
}

export function TrainingView({
  file,
  loadingTraining,
  onStartTraining,
  result,
  selectedDataset,
  stats,
  taskError,
  taskStatus,
  taskStatusAt,
}: TrainingViewProps) {
  const { t } = useTranslation();
  const [options, setOptions] = useState<TrainingOptions | null>(null);
  const [modelType, setModelType] = useState<TrainingModelTypeId>("ml");
  const [modelName, setModelName] = useState("");
  const [eegParams, setEegParams] = useState<TrainingControlValues>({});
  const [modelParams, setModelParams] = useState<TrainingControlValues>({});
  const [trainingParams, setTrainingParams] = useState<TrainingControlValues>({});
  const [resultPatientFilter, setResultPatientFilter] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    void getTrainingOptions()
      .then((trainingOptions) => {
        if (cancelled) {
          return;
        }

        const defaultType = trainingOptions.default_model_type;
        const defaultModel = trainingOptions.default_models[defaultType];

        setOptions(trainingOptions);
        setModelType(defaultType);
        setModelName(defaultModel);
        setEegParams(trainingOptions.default_eeg_params[defaultType]);
        setModelParams(modelDefaults(trainingOptions, defaultType, defaultModel));
        setTrainingParams(trainingOptions.default_training_params);
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(
              caughtError,
              "errors.training.options",
            ),
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const currentModels = options?.model_types[modelType].models ?? {};
  const currentModel = currentModels[modelName];
  const currentModelParameters = currentModel?.parameters ?? {};

  const visibleTrainingParams = useMemo(() => {
    const allowed = options?.training_params_by_type[modelType] ?? [];
    return Object.entries(options?.training_params ?? {}).filter(([name]) =>
      allowed.includes(name),
    );
  }, [modelType, options]);

  const filteredPatientResults = useMemo(() => {
    if (!result) {
      return [];
    }

    const normalizedFilter = resultPatientFilter.toLowerCase();
    return result.patient_results.filter((patient) =>
      patient.patient_id.toLowerCase().includes(normalizedFilter),
    );
  }, [result, resultPatientFilter]);

  function handleModelTypeChange(nextType: TrainingModelTypeId): void {
    if (!options) {
      return;
    }

    const nextModel = options.default_models[nextType];
    setModelType(nextType);
    setModelName(nextModel);
    setModelParams(modelDefaults(options, nextType, nextModel));
    setEegParams(options.default_eeg_params[nextType]);
  }

  function handleModelNameChange(event: ChangeEvent<HTMLSelectElement>): void {
    if (!options) {
      return;
    }

    const nextModel = event.target.value;
    setModelName(nextModel);
    setModelParams(modelDefaults(options, modelType, nextModel));
  }

  function updateEegParam(name: string, value: string): void {
    setEegParams((current) => ({ ...current, [name]: normalizeValue(value) }));
  }

  function updateModelParam(name: string, value: string): void {
    setModelParams((current) => ({ ...current, [name]: normalizeValue(value) }));
  }

  function updateTrainingParam(name: string, value: string): void {
    setTrainingParams((current) => ({
      ...current,
      [name]: normalizeValue(value),
    }));
  }

  async function handleRunTraining(): Promise<void> {
    if (!file && !selectedDataset) {
      setError(t("training.missingFile"));
      return;
    }

    setError("");
    await onStartTraining(file, {
      datasetId: selectedDataset?.id,
      modelType,
      modelName,
      eegParams,
      modelParams,
      trainingParams,
    });
  }

  return (
    <section className="training-layout interactive-training">
      {error && <div className="alert alert-error">{error}</div>}
      {taskError && <div className="alert alert-error">{taskError}</div>}

      {!file && !selectedDataset && (
        <div className="panel">
          <p className="muted">
            {t("training.uploadFirst")} <strong>{t("tabs.dataset")}</strong>.
          </p>
        </div>
      )}

      {(file || selectedDataset) && !stats && (
        <div className="panel">
          <p className="muted">
            <Trans
              i18nKey="training.datasetNotAnalyzed"
              values={{ file: file?.name || selectedDataset?.filename }}
              components={{ strong: <strong /> }}
            />
          </p>
        </div>
      )}

      <TrainingEegParamsPanel
        eegParams={eegParams}
        modelType={modelType}
        onEegParamChange={updateEegParam}
        options={options}
      />

      <TrainingModelPanel
        currentModelParameters={currentModelParameters}
        currentModels={currentModels}
        modelName={modelName}
        modelParams={modelParams}
        modelType={modelType}
        onModelNameChange={handleModelNameChange}
        onModelParamChange={updateModelParam}
        onModelTypeChange={handleModelTypeChange}
        onTrainingParamChange={updateTrainingParam}
        trainingParams={trainingParams}
        visibleTrainingParams={visibleTrainingParams}
      />

      {/* Hermana de .training-layout, no hija del panel: sticky se recorta a la
          caja del padre, y dentro del panel dejaria de pegarse al pasarlo. */}
      <TrainingActionBar
        datasetName={file?.name ?? selectedDataset?.filename}
        durationSeconds={result?.training_time_seconds}
        loadingTraining={loadingTraining}
        modelLabel={currentModel?.display_name ?? modelName}
        onRunTraining={handleRunTraining}
        patients={stats?.n_patients}
        ready={Boolean(file || selectedDataset)}
        trainingStatus={taskStatus}
        trainingStatusAt={taskStatusAt}
      />

      <TrainingResultsPanel
        filteredPatientResults={filteredPatientResults}
        onPatientFilterChange={(event) => {
          setResultPatientFilter(event.target.value);
        }}
        patientFilter={resultPatientFilter}
        result={result}
      />
    </section>
  );
}
