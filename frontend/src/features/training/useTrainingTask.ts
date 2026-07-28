import { useCallback, useEffect, useRef, useState } from "react";

import { waitForTaskResult } from "../../shared/api/tasks";
import type { TaskStatus } from "../../shared/types";
import { runTraining } from "./api";
import type {
  TrainingPayload,
  TrainingResult,
  TrainingTaskStatus,
} from "./types";
import { errorMessage, translate } from "../../shared/utils/errors";

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
  statusAt: Date | null;
  trainingInProgress: boolean;
}

export function useTrainingTask(
  onSuccess?: (result: TrainingResult) => void,
): UseTrainingTaskResult {
  const [taskId, setTaskId] = useState<string | null>(() =>
    window.sessionStorage.getItem(TASK_STORAGE_KEY),
  );
  const [status, setStatus] = useState<TrainingTaskStatus>(null);
  const [statusAt, setStatusAt] = useState<Date | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [error, setError] = useState("");
  const onSuccessRef = useRef(onSuccess);
  // Avoid updating the timestamp when polling repeats the same status.
  const statusRef = useRef<TrainingTaskStatus>(null);

  const applyStatus = useCallback((next: TrainingTaskStatus): void => {
    if (statusRef.current === next) {
      return;
    }

    statusRef.current = next;
    setStatus(next);
    setStatusAt(new Date());
  }, []);

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
      failureMessage: translate("errors.training.failed"),
      missingResultMessage: translate("errors.training.empty"),
      onPollError: (caughtError) => {
        if (!controller.signal.aborted) {
          setError(
            errorMessage(
              caughtError,
              "errors.training.statusCheck",
            ),
          );
        }
      },
      onStatus: (task) => {
        if (controller.signal.aborted) {
          return;
        }

        applyStatus(task.status);
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
              "errors.training.failed",
            ),
          );
        }
      });

    return () => {
      controller.abort();
    };
  }, [applyStatus, taskId]);

  const startTraining = useCallback(
    async (
      file: File | null | undefined,
      payload: TrainingPayload,
    ): Promise<void> => {
      window.sessionStorage.removeItem(TASK_STORAGE_KEY);
      setTaskId(null);
      applyStatus("SUBMITTING");
      setResult(null);
      setError("");

      try {
        const task = await runTraining(file, payload);
        window.sessionStorage.setItem(TASK_STORAGE_KEY, task.task_id);
        setTaskId(task.task_id);
        applyStatus(task.status);
      } catch (caughtError) {
        applyStatus("FAILURE");
        setError(
          errorMessage(caughtError, "errors.training.start"),
        );
      }
    },
    [applyStatus],
  );

  const trainingInProgress =
    status === "SUBMITTING" ||
    Boolean(taskId && (status === null || !TERMINAL_STATUSES.has(status)));

  return {
    error,
    result,
    startTraining,
    status,
    statusAt,
    trainingInProgress,
  };
}
