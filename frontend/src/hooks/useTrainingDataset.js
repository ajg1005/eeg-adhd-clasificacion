import { useEffect, useState } from "react";

import {
  getSavedTrainingDatasets,
  getTaskStatus,
  startDatasetAnalysis,
  uploadTrainingDataset,
} from "../api";

const TASK_POLL_INTERVAL_MS = 1000;

async function analyzeSavedDataset(datasetId) {
  const { task_id: taskId } = await startDatasetAnalysis(datasetId);

  while (true) {
    const task = await getTaskStatus(taskId);

    if (task.status === "SUCCESS") {
      return task.result;
    }

    if (task.status === "FAILURE") {
      throw new Error(task.error || "No se pudo analizar el dataset");
    }

    await new Promise((resolve) => setTimeout(resolve, TASK_POLL_INTERVAL_MS));
  }
}


// Estado compartido del dataset entre "Dataset entrenamiento" y "Entrenamiento".
export function useTrainingDataset() {
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [savedDatasets, setSavedDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [classFilter, setClassFilter] = useState("all");
  const [maxPatients, setMaxPatients] = useState(10);
  const [loadingStats, setLoadingStats] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    refreshSavedDatasets();
  }, []);

  async function refreshSavedDatasets() {
    setLoadingDatasets(true);
    try {
      setSavedDatasets(await getSavedTrainingDatasets());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingDatasets(false);
    }
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files[0] || null;
    setFile(selectedFile);
    setSelectedDataset(null);
    setStats(null);
    setError("");
  }

  async function handleSavedDatasetChange(event) {
    const datasetId = Number(event.target.value);
    const dataset = savedDatasets.find((item) => item.id === datasetId) || null;

    setFile(null);
    setSelectedDataset(dataset);
    setStats(null);
    setError("");

    if (!dataset) {
      return;
    }

    setLoadingStats(true);
    try {
      setStats(await analyzeSavedDataset(dataset.id));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingStats(false);
    }
  }

  async function handleAnalyzeDataset() {
    if (!file && !selectedDataset) {
      setError("Sube primero un CSV EEG.");
      return;
    }

    setLoadingStats(true);
    setError("");

    try {
      if (selectedDataset) {
        setStats(await analyzeSavedDataset(selectedDataset.id));
      } else {
        const saved = await uploadTrainingDataset(file);
        setSelectedDataset(saved);
        setStats(await analyzeSavedDataset(saved.id));
        await refreshSavedDatasets();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingStats(false);
    }
  }

  function handleClassFilterChange(event) {
    setClassFilter(event.target.value);
  }

  function handleMaxPatientsChange(event) {
    setMaxPatients(Number(event.target.value));
  }

  return {
    file,
    stats,
    savedDatasets,
    selectedDataset,
    classFilter,
    maxPatients,
    loadingStats,
    loadingDatasets,
    error,
    setError,
    handleFileChange,
    handleSavedDatasetChange,
    handleAnalyzeDataset,
    handleClassFilterChange,
    handleMaxPatientsChange,
  };
}
