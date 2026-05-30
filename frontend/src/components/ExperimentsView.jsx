import { useCallback, useEffect, useMemo, useState } from "react";

import { getExperimentDetail, getExperiments } from "../api";
import { formatMetric } from "../utils/formatters";

function formatDate(value) {
  if (!value) {
    return "N/A";
  }

  return new Date(value).toLocaleString("es-ES");
}

export function ExperimentsView() {
  const [experiments, setExperiments] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");

  const loadExperiments = useCallback(async () => {
    await Promise.resolve();
    setLoadingList(true);
    setError("");

    try {
      const items = await getExperiments();
      setExperiments(items);
      setSelectedId((current) => current ?? items[0]?.id ?? null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    const timeoutId = window.setTimeout(loadExperiments, 0);
    return () => window.clearTimeout(timeoutId);
  }, [loadExperiments]);

  useEffect(() => {
    if (!selectedId) {
      return;
    }

    async function loadDetail() {
      await Promise.resolve();
      setLoadingDetail(true);
      setError("");

      try {
        setSelectedExperiment(await getExperimentDetail(selectedId));
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingDetail(false);
      }
    }

    loadDetail();
  }, [selectedId]);

  const selectedSummary = useMemo(
    () => experiments.find((experiment) => experiment.id === selectedId),
    [experiments, selectedId],
  );

  return (
    <section className="training-layout">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="panel">
        <div className="section-heading-row">
          <div>
            <h2>Experimentos guardados</h2>
            <p className="muted">Primer listado sencillo de los entrenamientos que se han guardado en la base de datos.</p>
          </div>
          <button
            className="primary-button compact-button"
            disabled={loadingList}
            onClick={loadExperiments}
            type="button"
          >
            {loadingList ? "Cargando..." : "Refrescar"}
          </button>
        </div>

        {experiments.length === 0 ? (
          <p className="muted">
            {loadingList ? "Cargando experimentos..." : "Todavia no hay experimentos guardados."}
          </p>
        ) : (
          <div className="patient-table-wrap">
            <table className="patient-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Fecha</th>
                  <th>Modelo</th>
                  <th>Dataset</th>
                  <th>F1</th>
                  <th>Balanced</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((experiment) => (
                  <tr
                    className={experiment.id === selectedId ? "selected-row" : ""}
                    key={experiment.id}
                    onClick={() => setSelectedId(experiment.id)}
                  >
                    <td>#{experiment.id}</td>
                    <td>{formatDate(experiment.created_at)}</td>
                    <td>{experiment.model_type} / {experiment.model_name}</td>
                    <td>{experiment.dataset.filename}</td>
                    <td>{formatMetric(experiment.f1_score)}</td>
                    <td>{formatMetric(experiment.balanced_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {(selectedSummary || loadingDetail) && (
        <div className="panel">
          <div className="section-heading-row">
            <h2>Detalle</h2>
            {selectedId && <span className="importance-meta">Experimento #{selectedId}</span>}
          </div>

          {loadingDetail && <p className="muted">Cargando detalle...</p>}

          {selectedExperiment && (
            <>
              <div className="metric-grid metrics-wide">
                <div>
                  <span>Accuracy</span>
                  <strong>{formatMetric(selectedExperiment.accuracy)}</strong>
                </div>
                <div>
                  <span>Precision</span>
                  <strong>{formatMetric(selectedExperiment.precision)}</strong>
                </div>
                <div>
                  <span>Recall</span>
                  <strong>{formatMetric(selectedExperiment.recall)}</strong>
                </div>
                <div>
                  <span>F1</span>
                  <strong>{formatMetric(selectedExperiment.f1_score)}</strong>
                </div>
                <div>
                  <span>Tiempo</span>
                  <strong>{selectedExperiment.training_time_seconds.toFixed(2)}s</strong>
                </div>
              </div>

              <h3>Dataset</h3>
              <p className="muted">
                {selectedExperiment.dataset.filename} - {selectedExperiment.dataset.rows} filas - {selectedExperiment.dataset.n_subjects} pacientes
              </p>

              <h3>Configuracion</h3>
              <pre className="training-log">
                {JSON.stringify(
                  {
                    eeg_params: selectedExperiment.eeg_params,
                    model_params: selectedExperiment.model_params,
                    training_params: selectedExperiment.training_params,
                  },
                  null,
                  2,
                )}
              </pre>
            </>
          )}
        </div>
      )}
    </section>
  );
}
