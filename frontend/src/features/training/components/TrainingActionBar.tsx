import { useTranslation } from "react-i18next";

import type { TrainingTaskStatus } from "../types";

interface TrainingActionBarProps {
  datasetName: string | undefined;
  loadingTraining: boolean;
  modelLabel: string;
  onRunTraining: () => Promise<void>;
  patients: number | undefined;
  ready: boolean;
  trainingStatus: TrainingTaskStatus;
}

// Fija al pie: el boton de entrenar estaba al final de una rejilla de parametros
// de altura variable y se perdia de vista justo cuando hacia falta. Lleva al lado
// el resumen de lo que se va a ejecutar.
export function TrainingActionBar({
  datasetName,
  loadingTraining,
  modelLabel,
  onRunTraining,
  patients,
  ready,
  trainingStatus,
}: TrainingActionBarProps) {
  const { t } = useTranslation();
  const trainingStatusLabel = trainingStatus
    ? t(`training.taskStatuses.${trainingStatus}`, {
        defaultValue: trainingStatus,
      })
    : "";

  return (
    <div className="training-action-bar">
      <div className="training-action-summary">
        {datasetName && (
          <span>
            <span className="muted">{t("common.dataset")}</span> {datasetName}
          </span>
        )}
        {patients !== undefined && (
          <span>
            <span className="muted">{t("common.patients")}</span> {patients}
          </span>
        )}
        {modelLabel && (
          <span>
            <span className="muted">{t("common.model")}</span> {modelLabel}
          </span>
        )}
      </div>

      <div className="training-action-controls">
        {loadingTraining && (
          <span className="muted">
            {t("training.taskStatus", { status: trainingStatusLabel })}
          </span>
        )}
        <button
          className="primary-button"
          disabled={!ready || loadingTraining}
          onClick={() => {
            void onRunTraining();
          }}
          type="button"
        >
          {loadingTraining ? t("training.training") : t("training.train")}
        </button>
      </div>
    </div>
  );
}
