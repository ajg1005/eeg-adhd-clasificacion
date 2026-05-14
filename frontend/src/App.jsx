import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getHealth,
  getModelInfo,
  getModels,
  getSignalPreview,
  predictCsv,
  validateCsv,
  getModelFigures,
} from "./api";
import "./App.css";

const TABS = ["Datos", "Modelo", "Señal EEG", "Predicción"];

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatMetric(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toFixed(3);
}

function App() {
  const [activeTab, setActiveTab] = useState("Datos");
  const [apiStatus, setApiStatus] = useState("checking");
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("ml_best");
  const [modelInfo, setModelInfo] = useState(null);
  const [file, setFile] = useState(null);
  const [validation, setValidation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [signalPreview, setSignalPreview] = useState(null);
  const [selectedChannel, setSelectedChannel] = useState("Fp1");
  const [maxPoints, setMaxPoints] = useState(1000);
  const [error, setError] = useState("");
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [modelFigures, setModelFigures] = useState([]);


  useEffect(() => {
    async function loadInitialData() {
      try {
        await getHealth();
        setApiStatus("ok");

        const availableModels = await getModels();
        setModels(availableModels);

        const info = await getModelInfo(selectedModelId);
        setModelInfo(info);
        
        const figures = await getModelFigures(selectedModelId);
        setModelFigures(figures);

        if (info.channels?.length > 0) {
          setSelectedChannel(info.channels[0]);
        }
      } catch (err) {
        setApiStatus("error");
        setError(err.message);
      }
    }

    loadInitialData();
  }, [selectedModelId]);

  function handleModelChange(event) {
    setSelectedModelId(event.target.value);
    setModelInfo(null);
    setPrediction(null);
    setModelFigures([]);
    setError("");
  }

  async function handleFileChange(event) {
    const selectedFile = event.target.files[0];

    setFile(selectedFile);
    setValidation(null);
    setPrediction(null);
    setSignalPreview(null);
    setError("");

    if (!selectedFile) {
      return;
    }

    setLoadingValidation(true);

    try {
      const result = await validateCsv(selectedFile);
      setValidation(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingValidation(false);
    }
  }

  async function handleLoadPreview() {
    if (!file) {
      setError("Primero sube un archivo CSV.");
      return;
    }

    setLoadingPreview(true);
    setError("");

    try {
      const result = await getSignalPreview(file, selectedChannel, maxPoints);
      setSignalPreview(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function handlePrediction() {
    if (!file) {
      setError("Primero sube un archivo CSV.");
      return;
    }

    setLoadingPrediction(true);
    setError("");

    try {
      const result = await predictCsv(file, selectedModelId);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPrediction(false);
    }
  }

  const metrics = modelInfo?.metrics?.cv_metrics || modelInfo?.metrics;

  const predictionChartData = useMemo(() => {
    if (!prediction) {
      return [];
    }

    return Object.entries(prediction.epoch_percentage_by_class).map(
      ([label, percentage]) => ({
        clase: label,
        porcentaje: Number((percentage * 100).toFixed(2)),
        epochs: prediction.epoch_count_by_class[label] || 0,
      })
    );
  }, [prediction]);

  const decisionScore = prediction
    ? prediction.decision_score ?? prediction.confidence
    : null;

  const finalClassEpochPercentage = prediction
    ? prediction.final_class_epoch_percentage ??
      prediction.epoch_percentage_by_class[prediction.prediction_label]
    : null;

  const adhdEpochs = prediction?.epoch_count_by_class?.ADHD ?? 0;
  const controlEpochs = prediction?.epoch_count_by_class?.Control ?? 0;
  const thresholdUsed =
    prediction?.threshold !== undefined && prediction?.threshold !== null
      ? Number(prediction.threshold).toFixed(3)
      : "N/A";

  const metricsChartData = useMemo(() => {
    if (!metrics) {
      return [];
    }

    return [
      { name: "Accuracy", value: metrics.accuracy_epoch_mean },
      { name: "Balanced", value: metrics.balanced_accuracy_epoch_mean },
      { name: "Precision", value: metrics.precision_epoch_mean },
      { name: "Recall", value: metrics.recall_epoch_mean },
      { name: "F1", value: metrics.f1_epoch_mean },
    ].map((item) => ({
      ...item,
      value: Number((item.value || 0).toFixed(3)),
    }));
  }, [metrics]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">EEG ADHD Classifier</p>
          <h1>Herramienta experimental de apoyo al análisis de TDAH mediante EEG</h1>
          <p className="subtitle">
            Aplicación para validar señales EEG, visualizar
            canales y ejecutar inferencia con el mejor modelo de deep learning o machine learning.
          </p>
        </div>

        <div className={`api-pill api-pill-${apiStatus}`}>
          API: {apiStatus === "ok" ? "online" : apiStatus}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            className={activeTab === tab ? "tab-button active" : "tab-button"}
            key={tab}
            onClick={() => setActiveTab(tab)}
            type="button"
          >
            {tab}
          </button>
        ))}
      </nav>

      <section className="model-selector panel">
        <label>
          Modelo de inferencia
          <select value={selectedModelId} onChange={handleModelChange}>
            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.display_name}
              </option>
            ))}
          </select>
        </label>

        {modelInfo && (
          <p className="muted">
            {modelInfo.display_name} · {modelInfo.model_name} ·{" "}
            {modelInfo.model_family}
          </p>
        )}
      </section>

      {error && <div className="alert alert-error">{error}</div>}

      {activeTab === "Datos" && (
        <section className="grid-layout">
          <div className="panel">
            <h2>Archivo EEG</h2>

            <label className="file-drop">
              <span>Seleccionar CSV</span>
              <input type="file" accept=".csv" onChange={handleFileChange} />
            </label>

            {file && (
              <div className="file-info">
                <strong>{file.name}</strong>
                <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
            )}

            {loadingValidation && <p>Validando archivo...</p>}

            {validation && (
              <div className="validation-box">
                <p className="ok-text">CSV valido</p>
                <div className="metric-grid">
                  <div>
                    <span>Filas</span>
                    <strong>{validation.rows}</strong>
                  </div>
                  <div>
                    <span>Columnas</span>
                    <strong>{validation.columns}</strong>
                  </div>
                  <div>
                    <span>Canales</span>
                    <strong>{validation.available_channels.length}</strong>
                  </div>
                  <div>
                    <span>ID</span>
                    <strong>{validation.has_id ? "Si" : "No"}</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="panel">
            <h2>Validación de canales</h2>

            {modelInfo ? (
              <>
                <p className="muted">
                  El modelo espera {modelInfo.channels.length} canales EEG.
                </p>
                <div className="channel-list">
                  {modelInfo.channels.map((channel) => (
                    <span
                      className={
                        validation?.available_channels?.includes(channel)
                          ? "channel-ok"
                          : ""
                      }
                      key={channel}
                    >
                      {channel}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <p>Cargando canales esperados...</p>
            )}
          </div>
        </section>
      )}

{activeTab === "Modelo" && (
  <>
    <section className="grid-layout">
      <div className="panel">
        <h2>Modelo entrenado</h2>

        {modelInfo ? (
          <div className="metric-grid">
            <div>
              <span>Modelo</span>
              <strong>{modelInfo.model_name}</strong>
            </div>
            <div>
              <span>Features</span>
              <strong>{modelInfo.feature_mode}</strong>
            </div>
            <div>
              <span>Frecuencia</span>
              <strong>{modelInfo.sfreq} Hz</strong>
            </div>
            <div>
              <span>Epoch size</span>
              <strong>{modelInfo.epoch_size}</strong>
            </div>
            <div>
              <span>Step size</span>
              <strong>{modelInfo.step_size}</strong>
            </div>
                <div>
                  <span>N features</span>
                  <strong>{modelInfo.n_features ?? "N/A"}</strong>
                </div>
          </div>
        ) : (
          <p>Cargando información del modelo...</p>
        )}
      </div>

      <div className="panel">
        <h2>Métricas CV</h2>

        {metrics ? (
          <>
            <div className="chart-box">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metricsChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#116a7b" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="metric-grid metrics-wide">
              <div>
                <span>Accuracy</span>
                <strong>{formatMetric(metrics.accuracy_epoch_mean)}</strong>
              </div>
              <div>
                <span>Balanced Acc.</span>
                <strong>{formatMetric(metrics.balanced_accuracy_epoch_mean)}</strong>
              </div>
              <div>
                <span>Precision</span>
                <strong>{formatMetric(metrics.precision_epoch_mean)}</strong>
              </div>
              <div>
                <span>Recall</span>
                <strong>{formatMetric(metrics.recall_epoch_mean)}</strong>
              </div>
              <div>
                <span>F1</span>
                <strong>{formatMetric(metrics.f1_epoch_mean)}</strong>
              </div>
            </div>
          </>
        ) : (
          <p>No hay metricas disponibles.</p>
        )}
      </div>
    </section>

    {modelFigures.length > 0 && (
      <div className="panel figures-panel">
        <h2>Figuras del modelo</h2>

        <div className="figures-grid">
          {modelFigures.map((figure) => (
            <figure key={figure.url} className="model-figure">
              <img src={figure.url} alt={figure.title} />
              <figcaption>{figure.title}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    )}
  </>
)}


      {activeTab === "Señal EEG" && (
        <section className="panel">
          <h2>Visualización de señal EEG</h2>

          <div className="controls-row">
            <label>
              Canal
              <select
                value={selectedChannel}
                onChange={(event) => setSelectedChannel(event.target.value)}
              >
                {(modelInfo?.channels || []).map((channel) => (
                  <option key={channel} value={channel}>
                    {channel}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Muestras
              <input
                max="5000"
                min="250"
                step="250"
                type="number"
                value={maxPoints}
                onChange={(event) => setMaxPoints(Number(event.target.value))}
              />
            </label>

            <button
              className="primary-button compact-button"
              disabled={!file || loadingPreview}
              onClick={handleLoadPreview}
              type="button"
            >
              {loadingPreview ? "Cargando..." : "Ver señal"}
            </button>
          </div>

          {!file && <p className="muted">Primero sube un CSV en la pestaña Datos.</p>}

          {signalPreview && (
            <div className="chart-box tall-chart">
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={signalPreview.samples}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    dataKey="value"
                    dot={false}
                    isAnimationActive={false}
                    stroke="#116a7b"
                    strokeWidth={2}
                    type="monotone"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      )}

      {activeTab === "Predicción" && (
        <section className="grid-layout">
          <div className="panel">
            <h2>Ejecutar predicción</h2>

            {file ? (
              <div className="file-info">
                <strong>{file.name}</strong>
                <span>{validation ? `${validation.rows} filas` : "CSV cargado"}</span>
              </div>
            ) : (
              <p className="muted">Primero sube un CSV en la pestaña Datos.</p>
            )}

            <button
              className="primary-button"
              disabled={!file || loadingPrediction}
              onClick={handlePrediction}
              type="button"
            >
              {loadingPrediction ? "Procesando..." : "Ejecutar predicción"}
            </button>
          </div>

          <div className="panel">
            <h2>Resultado</h2>

            {prediction ? (
              <>
                <div className="result-main">
                  <div>
                    <span>Resultado global</span>
                    <strong>{prediction.prediction_label}</strong>
                  </div>
                  <div>
                    <span>Score global</span>
                    <strong>{formatPercent(decisionScore)}</strong>
                  </div>
                  <div>
                    <span>Epochs analizadas</span>
                    <strong>{prediction.n_epochs}</strong>
                  </div>
                  <div>
                    <span>Epochs ADHD</span>
                    <strong>{adhdEpochs}</strong>
                  </div>
                  <div>
                    <span>Epochs Control</span>
                    <strong>{controlEpochs}</strong>
                  </div>
                  <div>
                    <span>Porcentaje {prediction.prediction_label}</span>
                    <strong>{formatPercent(finalClassEpochPercentage)}</strong>
                  </div>
                  <div>
                    <span>Threshold usado</span>
                    <strong>{thresholdUsed}</strong>
                  </div>
                </div>

                <h3>Distribucion por epoch</h3>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={predictionChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="clase" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Bar
                        dataKey="porcentaje"
                        fill="#116a7b"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            ) : (
              <p className="muted">Todavía no hay predicción ejecutada.</p>
            )}
          </div>
        </section>
      )}
    </main>
  );
}

export default App;
