import type { ChangeEventHandler } from "react";
import { useTranslation } from "react-i18next";

import type { SelectOption } from "../../../shared/types";
import type {
  TrainingControlValues,
  TrainingModelOption,
  TrainingModelTypeId,
  TrainingOptionValue,
  TrainingTaskStatus,
} from "../types";
import {
  modelParamLabel,
  optionValueLabel,
  trainingParamLabel,
} from "../trainingLabels";
import { ModelSelectField } from "../../../shared/components/ModelSelectField";

interface TrainingModelPanelProps {
  currentModelParameters: Record<string, TrainingOptionValue[]>;
  currentModels: Record<string, TrainingModelOption>;
  datasetSelected: boolean;
  file: File | null;
  loadingTraining: boolean;
  modelName: string;
  modelParams: TrainingControlValues;
  modelType: TrainingModelTypeId;
  onModelNameChange: ChangeEventHandler<HTMLSelectElement>;
  onModelParamChange: (name: string, value: string) => void;
  onModelTypeChange: (modelType: TrainingModelTypeId) => void;
  onRunTraining: () => Promise<void>;
  onTrainingParamChange: (name: string, value: string) => void;
  trainingParams: TrainingControlValues;
  trainingStatus: TrainingTaskStatus;
  visibleTrainingParams: [string, TrainingOptionValue[]][];
}

function selectValue(value: TrainingOptionValue | undefined): string {
  return value === null || value === undefined ? "none" : String(value);
}

export function TrainingModelPanel({
  currentModelParameters,
  currentModels,
  datasetSelected,
  file,
  loadingTraining,
  modelName,
  modelParams,
  modelType,
  onModelNameChange,
  onModelParamChange,
  onModelTypeChange,
  onRunTraining,
  onTrainingParamChange,
  trainingParams,
  trainingStatus,
  visibleTrainingParams,
}: TrainingModelPanelProps) {
  const { t } = useTranslation();
  const trainingStatusLabel = trainingStatus
    ? t(`training.taskStatuses.${trainingStatus}`, {
        defaultValue: trainingStatus,
      })
    : "";
  const modelOptions: SelectOption[] = Object.entries(currentModels).map(
    ([key, model]) => ({
      label: model.display_name,
      value: key,
    }),
  );

  return (
    <div className="panel">
      <h2>{t("common.model")}</h2>
      <div className="segmented-control">
        <button
          className={modelType === "ml" ? "active" : ""}
          onClick={() => {
            onModelTypeChange("ml");
          }}
          type="button"
        >
          Machine Learning
        </button>
        <button
          className={modelType === "dl" ? "active" : ""}
          onClick={() => {
            onModelTypeChange("dl");
          }}
          type="button"
        >
          Deep Learning
        </button>
      </div>

      <div className="controls-row">
        <ModelSelectField
          label={t("model.trainingSelector")}
          onChange={onModelNameChange}
          options={modelOptions}
          value={modelName}
        />

        {Object.entries(currentModelParameters).map(([name, values]) => (
          <label key={name}>
            {modelParamLabel(t, name)}
            <select
              onChange={(event) => {
                onModelParamChange(name, event.target.value);
              }}
              value={selectValue(modelParams[name])}
            >
              {values.map((value) => (
                <option
                  key={String(value)}
                  value={value === null ? "none" : String(value)}
                >
                  {optionValueLabel(t, value)}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      {visibleTrainingParams.length > 0 && (
        <>
          <h3>{t("training.optimization")}</h3>
          <div className="controls-row">
            {visibleTrainingParams.map(([name, values]) => (
              <label key={name}>
                {trainingParamLabel(t, name)}
                <select
                  onChange={(event) => {
                    onTrainingParamChange(name, event.target.value);
                  }}
                  value={String(trainingParams[name])}
                >
                  {values.map((value) => (
                    <option key={String(value)} value={String(value)}>
                      {optionValueLabel(t, value)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </>
      )}

      <button
        className="primary-button"
        disabled={(!file && !datasetSelected) || loadingTraining}
        onClick={() => {
          void onRunTraining();
        }}
        type="button"
      >
        {loadingTraining ? t("training.training") : t("training.train")}
      </button>

      {loadingTraining && (
        <p className="muted">
          {t("training.trainingHint")}{" "}
          {t("training.taskStatus", { status: trainingStatusLabel })}
        </p>
      )}
    </div>
  );
}
