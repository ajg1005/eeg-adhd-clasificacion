import { useCallback, useEffect, useRef, useState } from "react";

import { waitForTaskResult } from "../../shared/api/tasks";
import type { TaskStatus } from "../../shared/types";
import { runTraining } from "./api";
import type {
  TrainingPayload,
  TrainingResult,
  TrainingTaskStatus,
  TrainingTaskSummary,
} from "./types";
import { errorMessage, translate } from "../../shared/utils/errors";

const TASK_STORAGE_KEY = "eeg-adhd-training-task-id";
const SUMMARY_STORAGE_KEY = "eeg-adhd-training-task-summary";

function restoreTaskSummary(): TrainingTaskSummary | null {
  try {
    const raw: unknown = JSON.parse(window.sessionStorage.getItem(SUMMARY_STORAGE_KEY) ?? "null");
    if (
      !raw || typeof raw !== "object" ||
      !("taskId" in raw) || !raw.taskId ||
      raw.taskId !== window.sessionStorage.getItem(TASK_STORAGE_KEY) ||
      !("modelLabel" in raw) || typeof raw.modelLabel !== "string"
    ) return null;
    return {
      modelLabel: raw.modelLabel,
      datasetName: "datasetName" in raw && typeof raw.datasetName === "string" ? raw.datasetName : undefined,
      patients: "patients" in raw && typeof raw.patients === "number" ? raw.patients : undefined,
    };
  } catch {
    return null;
  }
}
const TERMINAL_STATUSES = new Set<TaskStatus>([
  "SUCCESS",
  "FAILURE",
  "REVOKED",
]);

interface UseTrainingTaskResult {
  error: string;
  result: TrainingResult | null;
  summary: TrainingTaskSummary | null;
  startTraining: (
    file: File | null | undefined,
    payload: TrainingPayload,
    summary: TrainingTaskSummary,
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
  const [summary, setSummary] = useState<TrainingTaskSummary | null>(restoreTaskSummary);
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
          window.sessionStorage.removeItem(SUMMARY_STORAGE_KEY);
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
      taskSummary: TrainingTaskSummary,
    ): Promise<void> => {
      window.sessionStorage.removeItem(TASK_STORAGE_KEY);
      window.sessionStorage.removeItem(SUMMARY_STORAGE_KEY);
      const submittedPayload = structuredClone(payload);
      const submittedSummary = { ...taskSummary };
      setSummary(submittedSummary);
      setTaskId(null);
      applyStatus("SUBMITTING");
      setResult(null);
      setError("");

      try {
        const task = await runTraining(file, submittedPayload);
        window.sessionStorage.setItem(
          SUMMARY_STORAGE_KEY,
          JSON.stringify({ ...submittedSummary, taskId: task.task_id }),
        );
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
    summary,
    startTraining,
    status,
    statusAt,
    trainingInProgress,
  };
}
