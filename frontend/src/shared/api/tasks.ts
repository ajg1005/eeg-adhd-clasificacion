import type { TaskStatusResponse } from "../types";
import { requestJson, uuidPathSegment } from "./client";

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
