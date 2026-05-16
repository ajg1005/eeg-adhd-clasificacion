function selectValue(value) {
  return value === null || value === undefined ? "none" : String(value);
}

export function TrainingModelPanel({
  currentModelParameters,
  currentModels,
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
  return (
    <div className="panel training-section">
      <h2>Modelo</h2>
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
          Modelo
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
            {name}
            <select
              onChange={(event) => onModelParamChange(name, event.target.value)}
              value={selectValue(modelParams[name])}
            >
              {values.map((value) => (
                <option
                  key={String(value)}
                  value={value === null ? "none" : String(value)}
                >
                  {value === null ? "none" : String(value)}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      {visibleTrainingParams.length > 0 && (
        <>
          <h3>Entrenamiento</h3>
          <div className="controls-row">
            {visibleTrainingParams.map(([name, values]) => (
              <label key={name}>
                {name}
                <select
                  onChange={(event) =>
                    onTrainingParamChange(name, event.target.value)
                  }
                  value={String(trainingParams[name])}
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
        </>
      )}

      <button
        className="primary-button"
        disabled={!file || loadingTraining}
        onClick={onRunTraining}
        type="button"
      >
        {loadingTraining ? "Entrenando..." : "Entrenar modelo"}
      </button>
    </div>
  );
}
