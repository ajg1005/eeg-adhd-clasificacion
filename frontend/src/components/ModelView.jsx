import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatMetric } from "../utils/formatters";

function ModelCatalogGroup({ models, title }) {
  if (!models?.length) {
    return null;
  }

  return (
    <div className="catalog-group">
      <h3>{title}</h3>
      <div className="model-catalog-grid">
        {models.map((model) => (
          <article className="model-card" key={model.model_id}>
            <div>
              <span className="model-family-label">{model.model_family}</span>
              <h4>{model.display_name}</h4>
              <p className="muted">{model.description}</p>
              <p className="model-use-case">{model.use_case}</p>
            </div>

            <table className="params-table">
              <thead>
                <tr>
                  <th>Parametro</th>
                  <th>Valor comun</th>
                  <th>Uso</th>
                </tr>
              </thead>
              <tbody>
                {model.common_parameters.map((parameter) => (
                  <tr key={parameter.name}>
                    <td>{parameter.name}</td>
                    <td>{parameter.default}</td>
                    <td>{parameter.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        ))}
      </div>
    </div>
  );
}

export function ModelView({
  metrics,
  metricsChartData,
  modelCatalog,
  modelFigures,
  modelInfo,
}) {
  return (
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
                    <Bar dataKey="value" fill="#be7c4d" radius={[6, 6, 0, 0]} />
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
            <p>No hay métricas disponibles.</p>
          )}
        </div>
      </section>

      <div className="panel model-catalog-panel">
        <h2>Modelos y parámetros habituales</h2>
        <p className="muted">
          Catálogo de modelos candidatos para entrenamiento y comparación.
        </p>

        <ModelCatalogGroup
          models={modelCatalog?.machine_learning}
          title="Modelos clásicos de Machine Learning"
        />
        <ModelCatalogGroup
          models={modelCatalog?.deep_learning}
          title="Modelos de Deep Learning"
        />
      </div>

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
  );
}

