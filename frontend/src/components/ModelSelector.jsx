export function ModelSelector({ modelInfo, models, onModelChange, selectedModelId }) {
  return (
    <section className="model-selector panel">
      <label>
        Modelo de inferencia
        <select value={selectedModelId} onChange={onModelChange}>
          {models.map((model) => (
            <option key={model.model_id} value={model.model_id}>
              {model.display_name}
            </option>
          ))}
        </select>
      </label>

      {modelInfo && (
        <p className="muted">
          {modelInfo.display_name} · {modelInfo.model_name} · {" "}
          {modelInfo.model_family}
        </p>
      )}
    </section>
  );
}

