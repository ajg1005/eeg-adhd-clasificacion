import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import {
  optionValueLabel,
  signalParamLabel,
} from "../../utils/trainingLabels";

export function TrainingEegParamsPanel({
  eegParams,
  modelType,
  onEegParamChange,
  options,
}) {
  const { t } = useTranslation();
  const visibleParamNames = options?.eeg_params_by_type?.[modelType]
    || Object.keys(options?.eeg_params || {});
  const visibleParams = Object.entries(options?.eeg_params || {}).filter(([name]) =>
    visibleParamNames.includes(name)
  );

  return (
    <div className="panel training-section">
      <h2>{t("training.paramsTitle")}</h2>
      <p className="muted">
        {modelType === "ml"
          ? t("training.mlParamsDescription")
          : t("training.dlParamsDescription")}
      </p>
      <div className="controls-row">
        {visibleParams.map(([name, values]) => (
          <label key={name}>
            {signalParamLabel(t, name)}
            <select
              onChange={(event) => onEegParamChange(name, event.target.value)}
              value={String(eegParams[name])}
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
    </div>
  );
}

TrainingEegParamsPanel.propTypes = {
  eegParams: PropTypes.object.isRequired,
  modelType: PropTypes.string.isRequired,
  onEegParamChange: PropTypes.func.isRequired,
  options: PropTypes.shape({
    eeg_params: PropTypes.object,
    eeg_params_by_type: PropTypes.object,
  }),
};
