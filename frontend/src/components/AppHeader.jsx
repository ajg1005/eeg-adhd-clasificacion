export function AppHeader({ apiStatus }) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">EEG ADHD Classifier</p>
        <h1>Herramienta experimental de apoyo al análisis de TDAH mediante EEG</h1>
        <p className="subtitle">
          Aplicación para validar señales EEG, visualizar canales y ejecutar
          inferencia con el mejor modelo de deep learning o machine learning.
        </p>
      </div>

      <div className={`api-pill api-pill-${apiStatus}`}>
        API: {apiStatus === "ok" ? "online" : apiStatus}
      </div>
    </header>
  );
}

