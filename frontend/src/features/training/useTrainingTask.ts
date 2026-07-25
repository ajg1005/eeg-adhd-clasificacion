import { useCallback, useEffect, useRef, useState } from "react";

import { waitForTaskResult } from "../../shared/api/tasks";
import type { TaskStatus } from "../../shared/types";
import { runTraining } from "./api";
import type {
  TrainingPayload,
  TrainingResult,
  TrainingTaskStatus,
} from "./types";

const TASK_STORAGE_KEY = "eeg-adhd-training-task-id";
const TERMINAL_STATUSES = new Set<TaskStatus>([
  "SUCCESS",
  "FAILURE",
  "REVOKED",
]);

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
    const controller = new AbortController();

    void waitForTaskResult<TrainingResult>(activeTaskId, {
      failureMessage: "No se pudo completar el entrenamiento",
      missingResultMessage:
        "El entrenamiento ha terminado sin devolver resultados",
      onPollError: (caughtError) => {
        if (!controller.signal.aborted) {
          setError(
            errorMessage(
              caughtError,
              "No se pudo consultar el estado del entrenamiento",
            ),
          );
        }
      },
      onStatus: (task) => {
        if (controller.signal.aborted) {
          return;
        }

        setStatus(task.status);
        setError("");

        if (TERMINAL_STATUSES.has(task.status)) {
          window.sessionStorage.removeItem(TASK_STORAGE_KEY);
        }
      },
      retryOnPollError: true,
      signal: controller.signal,
    })
      .then((trainingResult) => {
        if (!controller.signal.aborted) {
          setResult(trainingResult);
          onSuccessRef.current?.(trainingResult);
        }
      })
      .catch((caughtError: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            errorMessage(
              caughtError,
              "No se pudo completar el entrenamiento",
            ),
          );
        }
      });

    return () => {
      controller.abort();
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
