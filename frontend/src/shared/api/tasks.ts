import type { TaskStatusResponse } from "../types";
import { requestJson, uuidPathSegment } from "./client";

const DEFAULT_POLL_INTERVAL_MS = 1000;

interface WaitForTaskOptions<TResult> {
  failureMessage: string;
  intervalMs?: number;
  missingResultMessage: string;
  onPollError?: (error: unknown) => void;
  onStatus?: (task: TaskStatusResponse<TResult>) => void;
  retryOnPollError?: boolean;
  signal?: AbortSignal;
}

export function getTaskStatus<TResult = unknown>(
  taskId: string,
): Promise<TaskStatusResponse<TResult>> {
  const safeTaskId = uuidPathSegment(taskId);

  return requestJson<TaskStatusResponse<TResult>>(
    `/tasks/${safeTaskId}`,
    undefined,
    "No se pudo consultar la tarea",
  );
}

function wait(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort);
      resolve();
    }, milliseconds);

    function handleAbort(): void {
      window.clearTimeout(timeoutId);
      reject(new DOMException("Task polling cancelled", "AbortError"));
    }

    if (signal?.aborted) {
      handleAbort();
      return;
    }

    signal?.addEventListener("abort", handleAbort, { once: true });
  });
}

export async function waitForTaskResult<TResult>(
  taskId: string,
  {
    failureMessage,
    intervalMs = DEFAULT_POLL_INTERVAL_MS,
    missingResultMessage,
    onPollError,
    onStatus,
    retryOnPollError = false,
    signal,
  }: WaitForTaskOptions<TResult>,
): Promise<TResult> {
  const safeTaskId = uuidPathSegment(taskId);

  while (!signal?.aborted) {
    let task: TaskStatusResponse<TResult>;

    try {
      task = await getTaskStatus<TResult>(safeTaskId);
    } catch (error) {
      if (!retryOnPollError) {
        throw error;
      }

      onPollError?.(error);
      await wait(intervalMs, signal);
      continue;
    }

    onStatus?.(task);

    if (task.status === "SUCCESS") {
      if (task.result === undefined || task.result === null) {
        throw new Error(missingResultMessage);
      }

      return task.result;
    }

    if (task.status === "FAILURE" || task.status === "REVOKED") {
      throw new Error(task.error || failureMessage);
    }

    await wait(intervalMs, signal);
  }

  throw new DOMException("Task polling cancelled", "AbortError");
}
