import {
  useEffect,
  useRef,
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
import { errorMessage, translate } from "../../shared/utils/errors";

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
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  handleSavedDatasetChange: (
    event: ChangeEvent<HTMLSelectElement>,
  ) => Promise<void>;
  handleAnalyzeDataset: () => Promise<void>;
  onClassFilterChange: (value: string) => void;
  handleMaxPatientsChange: (event: ChangeEvent<HTMLInputElement>) => void;
}
type AnalysisSource =
  | { kind: "file"; file: File }
  | { kind: "saved"; dataset: SavedTrainingDataset };

async function analyzeSavedDataset(
  datasetId: number,
): Promise<TrainingDatasetStats> {
  const { task_id: taskId } = await startDatasetAnalysis(datasetId);

  return waitForTaskResult<TrainingDatasetStats>(taskId, {
    failureMessage: translate("errors.datasets.analyze"),
    missingResultMessage: translate("errors.datasets.analysisEmpty"),
  });
}
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
  const analysisRequestRef = useRef(0);

  async function refreshSavedDatasets(): Promise<void> {
    try {
      setSavedDatasets(await getSavedTrainingDatasets());
    } catch (caughtError) {
      setError(errorMessage(caughtError, "errors.datasets.list"));
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
          setError(errorMessage(caughtError, "errors.datasets.list"));
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

  async function runAnalysis(source: AnalysisSource): Promise<void> {
    const requestId = analysisRequestRef.current + 1;
    analysisRequestRef.current = requestId;
    const isCurrent = (): boolean => analysisRequestRef.current === requestId;

    setLoadingStats(true);
    setError("");

    try {
      let datasetId = source.kind === "saved" ? source.dataset.id : 0;

      if (source.kind === "file") {
        const saved = await uploadTrainingDataset(source.file);

        if (!isCurrent()) {
          return;
        }

        setSelectedDataset(saved);
        datasetId = saved.id;
      }

      const analyzed = await analyzeSavedDataset(datasetId);

      if (!isCurrent()) {
        return;
      }

      setStats(analyzed);

      if (source.kind === "file") {
        setLoadingDatasets(true);
        await refreshSavedDatasets();
      }
    } catch (caughtError) {
      if (isCurrent()) {
        setError(errorMessage(caughtError, "errors.datasets.analyze"));
      }
    } finally {
      if (isCurrent()) {
        setLoadingStats(false);
      }
    }
  }
  async function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const selectedFile = event.target.files?.[0] ?? null;

    setFile(selectedFile);
    setSelectedDataset(null);
    setStats(null);
    setError("");

    if (selectedFile) {
      await runAnalysis({ kind: "file", file: selectedFile });
    }
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

    if (dataset) {
      await runAnalysis({ kind: "saved", dataset });
    }
  }
  async function handleAnalyzeDataset(): Promise<void> {
    if (selectedDataset) {
      await runAnalysis({ kind: "saved", dataset: selectedDataset });
      return;
    }

    if (file) {
      await runAnalysis({ kind: "file", file });
      return;
    }

    setError(translate("errors.datasets.missingCsv"));
  }

  function onClassFilterChange(value: string): void {
    setClassFilter(value);
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
    onClassFilterChange,
    handleMaxPatientsChange,
  };
}
