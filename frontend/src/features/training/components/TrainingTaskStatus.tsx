import { useTranslation } from "react-i18next";

import type { TrainingTaskStatus as TaskStatusValue } from "../types";

interface TrainingTaskStatusProps {
  durationSeconds: number | undefined;
  status: TaskStatusValue;
  statusAt: Date | null;
}

const STATUS_VARIANT: Record<string, "pending" | "active" | "failed"> = {
  SUBMITTING: "pending",
  PENDING: "pending",
  RECEIVED: "pending",
  RETRY: "pending",
  STARTED: "active",
  SUCCESS: "active",
  FAILURE: "failed",
  REVOKED: "failed",
};

export function TrainingTaskStatus({
  durationSeconds,
  status,
  statusAt,
}: TrainingTaskStatusProps) {
  const { i18n, t } = useTranslation();

  if (!status) {
    return null;
  }

  const variant = STATUS_VARIANT[status] ?? "pending";
  const detail =
    status === "SUCCESS" && durationSeconds !== undefined
      ? `${durationSeconds.toFixed(1)} s`
      : statusAt?.toLocaleTimeString(
          i18n.resolvedLanguage === "en" ? "en-US" : "es-ES",
        );

  return (
    <span aria-live="polite" className="task-status" role="status">
      <span aria-hidden="true" className={`task-status-dot ${variant}`} />
      {t(`training.taskStatuses.${status}`, { defaultValue: status })}
      {detail && <span className="task-status-detail">{detail}</span>}
    </span>
  );
}
