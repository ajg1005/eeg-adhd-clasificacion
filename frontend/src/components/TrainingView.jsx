import { useEffect, useMemo, useState } from "react";

import { getDatasetStats, getTrainingOptions, runTraining } from "../api";
import { TrainingDatasetPanel } from "./training/TrainingDatasetPanel";
import { TrainingEegParamsPanel } from "./training/TrainingEegParamsPanel";
import { TrainingModelPanel } from "./training/TrainingModelPanel";
import { TrainingResultsPanel } from "./training/TrainingResultsPanel";

function normalizeValue(value) {
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

function modelDefaults(options, modelType, modelName) {
  return options?.model_types?.[modelType]?.models?.[modelName]?.default_params || {};
}

export function TrainingView() {
  const [options, setOptions] = useState(null);
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [modelType, setModelType] = useState("ml");
  const [modelName, setModelName] = useState("");
  const [eegParams, setEegParams] = useState({});
  const [modelParams, setModelParams] = useState({});
  const [trainingParams, setTrainingParams] = useState({});
  const [result, setResult] = useState(null);
  const [resultPatientFilter, setResultPatientFilter] = useState("");
  const [loadingStats, setLoadingStats] = useState(false);
  const [loadingTraining, setLoadingTraining] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadOptions() {
      try {
        const trainingOptions = await getTrainingOptions();
        const defaultType = trainingOptions.default_model_type;
        const defaultModel = trainingOptions.default_models[defaultType];

        setOptions(trainingOptions);
        setModelType(defaultType);
        setModelName(defaultModel);
        setEegParams(trainingOptions.default_eeg_params[defaultType]);
        setModelParams(modelDefaults(trainingOptions, defaultType, defaultModel));
        setTrainingParams(trainingOptions.default_training_params);
      } catch (err) {
        setError(err.message);
      }
    }

    loadOptions();
  }, []);

  const currentModels = options?.model_types?.[modelType]?.models || {};
  const currentModel = currentModels[modelName];
  const currentModelParameters = currentModel?.parameters || {};

  const visibleTrainingParams = useMemo(() => {
    const allowed = options?.training_params_by_type?.[modelType] || [];
    return Object.entries(options?.training_params || {}).filter(([name]) =>
      allowed.includes(name)
    );
  }, [modelType, options]);

  const filteredPatientResults = useMemo(() => {
    if (!result?.patient_results) {
      return [];
    }

    return result.patient_results.filter((patient) =>
      patient.patient_id.toLowerCase().includes(resultPatientFilter.toLowerCase())
    );
  }, [result, resultPatientFilter]);

  function handleFileChange(event) {
    const selectedFile = event.target.files[0] || null;
    setFile(selectedFile);
    setStats(null);
    setResult(null);
    setError("");
  }

  async function handleAnalyzeDataset() {
    if (!file) {
      setError("Sube primero un CSV EEG.");
      return;
    }

    setLoadingStats(true);
    setError("");

    try {
      setStats(await getDatasetStats(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingStats(false);
    }
  }

  function handleModelTypeChange(nextType) {
    if (!options) {
      return;
    }

    const nextModel = options.default_models[nextType];
    setModelType(nextType);
    setModelName(nextModel);
    setModelParams(modelDefaults(options, nextType, nextModel));
    setEegParams(options.default_eeg_params[nextType]);
  }

  function handleModelNameChange(event) {
    if (!options) {
      return;
    }

    const nextModel = event.target.value;
    setModelName(nextModel);
    setModelParams(modelDefaults(options, modelType, nextModel));
  }

  function updateEegParam(name, value) {
    setEegParams((current) => ({ ...current, [name]: normalizeValue(value) }));
  }

  function updateModelParam(name, value) {
    setModelParams((current) => ({ ...current, [name]: normalizeValue(value) }));
  }

  function updateTrainingParam(name, value) {
    setTrainingParams((current) => ({ ...current, [name]: normalizeValue(value) }));
  }

  async function handleRunTraining() {
    if (!file) {
      setError("Sube primero un CSV EEG.");
      return;
    }

    setLoadingTraining(true);
    setError("");
    setResult(null);

    try {
      const trainingResult = await runTraining(file, {
        modelType,
        modelName,
        eegParams,
        modelParams,
        trainingParams,
      });
      setResult(trainingResult);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingTraining(false);
    }
  }

  return (
    <section className="training-layout interactive-training">
      {error && <div className="alert alert-error">{error}</div>}

      <TrainingDatasetPanel
        file={file}
        loadingStats={loadingStats}
        onAnalyzeDataset={handleAnalyzeDataset}
        onFileChange={handleFileChange}
        stats={stats}
      />

      <TrainingEegParamsPanel
        eegParams={eegParams}
        modelType={modelType}
        onEegParamChange={updateEegParam}
        options={options}
      />

      <TrainingModelPanel
        currentModelParameters={currentModelParameters}
        currentModels={currentModels}
        file={file}
        loadingTraining={loadingTraining}
        modelName={modelName}
        modelParams={modelParams}
        modelType={modelType}
        onModelNameChange={handleModelNameChange}
        onModelParamChange={updateModelParam}
        onModelTypeChange={handleModelTypeChange}
        onRunTraining={handleRunTraining}
        onTrainingParamChange={updateTrainingParam}
        trainingParams={trainingParams}
        visibleTrainingParams={visibleTrainingParams}
      />

      <TrainingResultsPanel
        filteredPatientResults={filteredPatientResults}
        onPatientFilterChange={(event) =>
          setResultPatientFilter(event.target.value)
        }
        patientFilter={resultPatientFilter}
        result={result}
      />
    </section>
  );
}
