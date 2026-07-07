import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import { fileShape, modelOptionShape } from "../../propTypes";
import {
  modelParamLabel,
  optionValueLabel,
  trainingParamLabel,
} from "../../utils/trainingLabels";

function selectValue(value) {
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
  visibleTrainingParams,
}) {
  const { t } = useTranslation();

  return (
    <div className="panel training-section">
      <h2>{t("common.model")}</h2>
      <div className="segmented-control">
        <button
          className={modelType === "ml" ? "active" : ""}
          onClick={() => onModelTypeChange("ml")}
          type="button"
        >
          Machine Learning
        </button>
        <button
          className={modelType === "dl" ? "active" : ""}
          onClick={() => onModelTypeChange("dl")}
          type="button"
        >
          Deep Learning
        </button>
      </div>

      <div className="controls-row">
        <label>
          {t("common.model")}
          <select value={modelName} onChange={onModelNameChange}>
            {Object.entries(currentModels).map(([key, model]) => (
              <option key={key} value={key}>
                {model.display_name}
              </option>
            ))}
          </select>
        </label>

        {Object.entries(currentModelParameters).map(([name, values]) => (
          <label key={name}>
            {modelParamLabel(t, name)}
            <select
              onChange={(event) => onModelParamChange(name, event.target.value)}
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
                  onChange={(event) =>
                    onTrainingParamChange(name, event.target.value)
                  }
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
        onClick={onRunTraining}
        type="button"
      >
        {loadingTraining ? t("training.training") : t("training.train")}
      </button>

      {loadingTraining && <p className="muted">{t("training.trainingHint")}</p>}
    </div>
  );
}

TrainingModelPanel.propTypes = {
  currentModelParameters: PropTypes.object.isRequired,
  currentModels: PropTypes.objectOf(modelOptionShape).isRequired,
  datasetSelected: PropTypes.bool,
  file: fileShape,
  loadingTraining: PropTypes.bool.isRequired,
  modelName: PropTypes.string.isRequired,
  modelParams: PropTypes.object.isRequired,
  modelType: PropTypes.string.isRequired,
  onModelNameChange: PropTypes.func.isRequired,
  onModelParamChange: PropTypes.func.isRequired,
  onModelTypeChange: PropTypes.func.isRequired,
  onRunTraining: PropTypes.func.isRequired,
  onTrainingParamChange: PropTypes.func.isRequired,
  trainingParams: PropTypes.object.isRequired,
  visibleTrainingParams: PropTypes.arrayOf(
    PropTypes.arrayOf(
      PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.number,
        PropTypes.bool,
        PropTypes.array,
      ]),
    ),
  ).isRequired,
};
