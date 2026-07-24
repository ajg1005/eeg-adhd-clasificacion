import { useCallback, useEffect, useRef, useState } from "react";

import { getTaskStatus, runTraining } from "../../api";
import type {
  TaskStatus,
  TrainingPayload,
  TrainingResult,
} from "../../types";

const TASK_STORAGE_KEY = "eeg-adhd-training-task-id";
const TASK_POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = new Set<TaskStatus>(["SUCCESS", "FAILURE"]);

export type TrainingTaskStatus = TaskStatus | "SUBMITTING" | null;

interface UseTrainingTaskResult {
  error: string;
  result: TrainingResult | null;
  startTraining: (
    file: File | null | undefined,
    payload: TrainingPayload,
  ) => Promise<void>;
  status: TrainingTaskStatus;
  trainingInProgress: boolean;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

export function useTrainingTask(
  onSuccess?: (result: TrainingResult) => void,
): UseTrainingTaskResult {
  const [taskId, setTaskId] = useState<string | null>(() =>
    window.sessionStorage.getItem(TASK_STORAGE_KEY),
  );
  const [status, setStatus] = useState<TrainingTaskStatus>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [error, setError] = useState("");
  const onSuccessRef = useRef(onSuccess);

  useEffect(() => {
    onSuccessRef.current = onSuccess;
  }, [onSuccess]);

  useEffect(() => {
    if (!taskId) {
      return;
    }

    const activeTaskId = taskId;
    let cancelled = false;
    let timeoutId: number | undefined;

    async function pollTask() {
      try {
        const task = await getTaskStatus<TrainingResult>(activeTaskId);

        if (cancelled) {
          return;
        }

        setStatus(task.status);
        setError("");

        if (task.status === "SUCCESS") {
          window.sessionStorage.removeItem(TASK_STORAGE_KEY);

          if (!task.result) {
            setError("El entrenamiento ha terminado sin devolver resultados");
            return;
          }

          setResult(task.result);
          onSuccessRef.current?.(task.result);
          return;
        }

        if (task.status === "FAILURE") {
          window.sessionStorage.removeItem(TASK_STORAGE_KEY);
          setError(task.error || "No se pudo completar el entrenamiento");
          return;
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(
            errorMessage(
              caughtError,
              "No se pudo consultar el estado del entrenamiento",
            ),
          );
        }
      }

      if (!cancelled) {
        timeoutId = window.setTimeout(pollTask, TASK_POLL_INTERVAL_MS);
      }
    }

    void pollTask();

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [taskId]);

  const startTraining = useCallback(
    async (
      file: File | null | undefined,
      payload: TrainingPayload,
    ): Promise<void> => {
      window.sessionStorage.removeItem(TASK_STORAGE_KEY);
      setTaskId(null);
      setStatus("SUBMITTING");
      setResult(null);
      setError("");

      try {
        const task = await runTraining(file, payload);
        window.sessionStorage.setItem(TASK_STORAGE_KEY, task.task_id);
        setTaskId(task.task_id);
        setStatus(task.status);
      } catch (caughtError) {
        setStatus("FAILURE");
        setError(
          errorMessage(caughtError, "No se pudo iniciar el entrenamiento"),
        );
      }
    },
    [],
  );

  const trainingInProgress =
    status === "SUBMITTING" ||
    Boolean(taskId && (status === null || !TERMINAL_STATUSES.has(status)));

  return {
    error,
    result,
    startTraining,
    status,
    trainingInProgress,
  };
}
