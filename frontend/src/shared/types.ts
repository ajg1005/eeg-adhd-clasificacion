export type UnknownRecord = Record<string, unknown>;
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { [key: string]: JsonValue };

export type TaskStatus =
  | "PENDING"
  | "RECEIVED"
  | "STARTED"
  | "RETRY"
  | "SUCCESS"
  | "FAILURE"
  | "REVOKED";

export interface AsyncTaskResponse {
  task_id: string;
  status: TaskStatus;
}

export interface TaskStatusResponse<TResult = unknown>
  extends AsyncTaskResponse {
  result?: TResult;
  error?: string;
}

export interface SelectOption {
  disabled?: boolean;
  label: string;
  value: string;
}
