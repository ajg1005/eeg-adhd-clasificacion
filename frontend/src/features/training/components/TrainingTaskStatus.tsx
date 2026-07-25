import { useTranslation } from "react-i18next";

import type { TrainingTaskStatus as TaskStatusValue } from "../types";

interface TrainingTaskStatusProps {
  // Segundos que ha tardado el entrenamiento, solo cuando ya hay resultado.
  durationSeconds: number | undefined;
  status: TaskStatusValue;
  statusAt: Date | null;
}

// La paleta del proyecto no tiene verde: plata para lo que espera, cobre para lo
// que avanza o ha terminado, y rosy-copper reservado a los fallos.
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
  const { t } = useTranslation();

  if (!status) {
    return null;
  }

  const variant = STATUS_VARIANT[status] ?? "pending";
  // Al terminar interesa cuanto ha tardado; mientras corre, desde cuando.
  const detail =
    status === "SUCCESS" && durationSeconds !== undefined
      ? `${durationSeconds.toFixed(1)} s`
      : statusAt?.toLocaleTimeString();

  return (
    <span className="task-status">
      <span className={`task-status-dot ${variant}`} />
      {t(`training.taskStatuses.${status}`, { defaultValue: status })}
      {detail && <span className="task-status-detail">{detail}</span>}
    </span>
  );
}
