import { useCallback, useEffect, useRef, useState } from "react";

import { getTaskStatus, runTraining } from "../api";

const TASK_STORAGE_KEY = "eeg-adhd-training-task-id";
const TASK_POLL_INTERVAL_MS = 1000;
const TERMINAL_STATUSES = new Set(["SUCCESS", "FAILURE"]);

export function useTrainingTask(onSuccess) {
  const [taskId, setTaskId] = useState(() =>
    window.sessionStorage.getItem(TASK_STORAGE_KEY),
  );
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const onSuccessRef = useRef(onSuccess);

  useEffect(() => {
    onSuccessRef.current = onSuccess;
  }, [onSuccess]);

  useEffect(() => {
    if (!taskId) {
      return undefined;
    }

    let cancelled = false;
    let timeoutId;

    async function pollTask() {
      try {
        const task = await getTaskStatus(taskId);

        if (cancelled) {
          return;
        }

        setStatus(task.status);
        setError("");

        if (task.status === "SUCCESS") {
          window.sessionStorage.removeItem(TASK_STORAGE_KEY);
          setResult(task.result);
          onSuccessRef.current?.(task.result);
          return;
        }

        if (task.status === "FAILURE") {
          window.sessionStorage.removeItem(TASK_STORAGE_KEY);
          setError(task.error || "No se pudo completar el entrenamiento");
          return;
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      }

      if (!cancelled) {
        timeoutId = window.setTimeout(pollTask, TASK_POLL_INTERVAL_MS);
      }
    }

    pollTask();

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [taskId]);

  const startTraining = useCallback(async (file, payload) => {
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
    } catch (err) {
      setStatus("FAILURE");
      setError(err.message);
    }
  }, []);

  const trainingInProgress =
    status === "SUBMITTING" ||
    Boolean(taskId && !TERMINAL_STATUSES.has(status));

  return {
    error,
    result,
    startTraining,
    status,
    trainingInProgress,
  };
}
