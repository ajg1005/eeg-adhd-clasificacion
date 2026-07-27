import { useTranslation } from "react-i18next";

import { TrainingTaskStatus } from "./TrainingTaskStatus";
import type { TrainingTaskStatus as TaskStatusValue } from "../types";

interface TrainingActionBarProps {
  datasetName: string | undefined;
  durationSeconds: number | undefined;
  loadingTraining: boolean;
  modelLabel: string;
  onRunTraining: () => Promise<void>;
  patients: number | undefined;
  ready: boolean;
  trainingStatus: TaskStatusValue;
  trainingStatusAt: Date | null;
}
export function TrainingActionBar({
  datasetName,
  durationSeconds,
  loadingTraining,
  modelLabel,
  onRunTraining,
  patients,
  ready,
  trainingStatus,
  trainingStatusAt,
}: TrainingActionBarProps) {
  const { t } = useTranslation();

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
        <TrainingTaskStatus
          durationSeconds={durationSeconds}
          status={trainingStatus}
          statusAt={trainingStatusAt}
        />
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
