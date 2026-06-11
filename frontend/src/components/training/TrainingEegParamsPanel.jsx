import PropTypes from "prop-types";

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
  const visibleParamNames = options?.eeg_params_by_type?.[modelType]
    || Object.keys(options?.eeg_params || {});
  const visibleParams = Object.entries(options?.eeg_params || {}).filter(([name]) =>
    visibleParamNames.includes(name)
  );

  return (
    <div className="panel training-section">
      <h2>Parámetros de entrenamiento</h2>
      <p className="muted">
        {modelType === "ml"
          ? "Configuración para segmentar la señal y extraer características antes de entrenar el modelo clásico."
          : "Configuración para preparar las ventanas de señal antes de entrenar el modelo de deep learning."}
      </p>
      <div className="controls-row">
        {visibleParams.map(([name, values]) => (
          <label key={name}>
            {signalParamLabel(name)}
            <select
              onChange={(event) => onEegParamChange(name, event.target.value)}
              value={String(eegParams[name])}
            >
              {values.map((value) => (
                <option key={String(value)} value={String(value)}>
                  {optionValueLabel(value)}
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
