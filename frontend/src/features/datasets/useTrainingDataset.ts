import {
  useEffect,
  useState,
  type ChangeEvent,
  type Dispatch,
  type SetStateAction,
} from "react";

import {
  getSavedTrainingDatasets,

  startDatasetAnalysis,
  uploadTrainingDataset,
} from "./api";
import { waitForTaskResult } from "../../shared/api/tasks";
import type {
  SavedTrainingDataset,
  TrainingDatasetStats,
} from "./types";

interface UseTrainingDatasetResult {
  file: File | null;
  stats: TrainingDatasetStats | null;
  savedDatasets: SavedTrainingDataset[];
  selectedDataset: SavedTrainingDataset | null;
  classFilter: string;
  maxPatients: number;
  loadingStats: boolean;
  loadingDatasets: boolean;
  error: string;
  setError: Dispatch<SetStateAction<string>>;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  handleSavedDatasetChange: (
    event: ChangeEvent<HTMLSelectElement>,
  ) => Promise<void>;
  handleAnalyzeDataset: () => Promise<void>;
  handleClassFilterChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  handleMaxPatientsChange: (event: ChangeEvent<HTMLInputElement>) => void;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

async function analyzeSavedDataset(
  datasetId: number,
): Promise<TrainingDatasetStats> {
  const { task_id: taskId } = await startDatasetAnalysis(datasetId);

  return waitForTaskResult<TrainingDatasetStats>(taskId, {
    failureMessage: "No se pudo analizar el dataset",
    missingResultMessage: "El análisis ha terminado sin devolver resultados",
  });
}
// Estado compartido del dataset entre "Dataset entrenamiento" y "Entrenamiento".
export function useTrainingDataset(): UseTrainingDatasetResult {
  const [file, setFile] = useState<File | null>(null);
  const [stats, setStats] = useState<TrainingDatasetStats | null>(null);
  const [savedDatasets, setSavedDatasets] = useState<SavedTrainingDataset[]>([]);
  const [selectedDataset, setSelectedDataset] =
    useState<SavedTrainingDataset | null>(null);
  const [classFilter, setClassFilter] = useState("all");
  const [maxPatients, setMaxPatients] = useState(10);
  const [loadingStats, setLoadingStats] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [error, setError] = useState("");

  async function refreshSavedDatasets(): Promise<void> {
    try {
      setSavedDatasets(await getSavedTrainingDatasets());
    } catch (caughtError) {
      setError(
        errorMessage(caughtError, "No se pudieron cargar los datasets guardados"),
      );
    } finally {
      setLoadingDatasets(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    void getSavedTrainingDatasets()
      .then((datasets) => {
        if (!cancelled) {
          setSavedDatasets(datasets);
        }
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(
              caughtError,
              "No se pudieron cargar los datasets guardados",
            ),
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingDatasets(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setSelectedDataset(null);
    setStats(null);
    setError("");
  }

  async function handleSavedDatasetChange(
    event: ChangeEvent<HTMLSelectElement>,
  ): Promise<void> {
    const datasetId = Number(event.target.value);
    const dataset =
      savedDatasets.find((item) => item.id === datasetId) ?? null;

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
    } catch (caughtError) {
      setError(errorMessage(caughtError, "No se pudo analizar el dataset"));
    } finally {
      setLoadingStats(false);
    }
  }

  async function handleAnalyzeDataset(): Promise<void> {
    if (!file && !selectedDataset) {
      setError("Sube primero un CSV EEG.");
      return;
    }

    setLoadingStats(true);
    setError("");

    try {
      if (selectedDataset) {
        setStats(await analyzeSavedDataset(selectedDataset.id));
      } else if (file) {
        const saved = await uploadTrainingDataset(file);
        setSelectedDataset(saved);
        setStats(await analyzeSavedDataset(saved.id));
        setLoadingDatasets(true);
        await refreshSavedDatasets();
      }
    } catch (caughtError) {
      setError(errorMessage(caughtError, "No se pudo analizar el dataset"));
    } finally {
      setLoadingStats(false);
    }
  }

  function handleClassFilterChange(
    event: ChangeEvent<HTMLSelectElement>,
  ): void {
    setClassFilter(event.target.value);
  }

  function handleMaxPatientsChange(
    event: ChangeEvent<HTMLInputElement>,
  ): void {
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
