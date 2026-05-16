export function TrainingEegParamsPanel({
  eegParams,
  modelType,
  onEegParamChange,
  options,
}) {
  return (
    <div className="panel training-section">
      <h2>Parametros EEG</h2>
      <p className="muted">
        {modelType === "ml"
          ? "ML usa la configuracion del script: epoch_size 1920, step_size 960 y nperseg 960."
          : "DL usa la configuracion del script: epoch_size 512, step_size 256, filtrado y zscore por paciente."}
      </p>
      <div className="controls-row">
        {Object.entries(options?.eeg_params || {}).map(([name, values]) => (
          <label key={name}>
            {name}
            <select
              onChange={(event) => onEegParamChange(name, event.target.value)}
              value={String(eegParams[name])}
            >
              {values.map((value) => (
                <option key={String(value)} value={String(value)}>
                  {String(value)}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </div>
  );
}
