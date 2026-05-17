import { useState } from "react";

import { getDatasetStats } from "../api";


// Estado compartido del dataset de entrenamiento entre las pestanas
// "Dataset entrenamiento" (carga y exploracion) y "Entrenamiento" (params + train).
export function useTrainingDataset() {
  const [file, setFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [classFilter, setClassFilter] = useState("all");
  const [maxPatients, setMaxPatients] = useState(10);
  const [loadingStats, setLoadingStats] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const selectedFile = event.target.files[0] || null;
    setFile(selectedFile);
    setStats(null);
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

  function handleClassFilterChange(event) {
    setClassFilter(event.target.value);
  }

  function handleMaxPatientsChange(event) {
    setMaxPatients(Number(event.target.value));
  }

  return {
    file,
    stats,
    classFilter,
    maxPatients,
    loadingStats,
    error,
    setError,
    handleFileChange,
    handleAnalyzeDataset,
    handleClassFilterChange,
    handleMaxPatientsChange,
  };
}
