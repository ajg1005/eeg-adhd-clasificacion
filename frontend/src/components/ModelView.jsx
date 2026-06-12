import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import { modelInfoShape } from "../propTypes";
import { formatMetric } from "../utils/formatters";

const catalogModelShape = PropTypes.shape({
  common_parameters: PropTypes.arrayOf(
    PropTypes.shape({
      default: PropTypes.string.isRequired,
      description: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired,
    }),
  ).isRequired,
  description: PropTypes.string.isRequired,
  display_name: PropTypes.string.isRequired,
  model_family: PropTypes.string.isRequired,
  model_id: PropTypes.string.isRequired,
  use_case: PropTypes.string.isRequired,
});

function ModelCatalogGroup({ models, title, t }) {
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
                  <th>{t("model.parameter")}</th>
                  <th>{t("model.commonValue")}</th>
                  <th>{t("model.usage")}</th>
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

function metricChartLabel(t, name) {
  const labels = {
    Accuracy: t("metrics.accuracy"),
    Balanced: t("metrics.balanced"),
    Precision: t("metrics.precision"),
    Recall: t("metrics.recall"),
    F1: t("metrics.f1"),
  };

  return labels[name] || name;
}

ModelCatalogGroup.propTypes = {
  models: PropTypes.arrayOf(catalogModelShape),
  t: PropTypes.func.isRequired,
  title: PropTypes.string.isRequired,
};

export function ModelView({
  metrics,
  metricsChartData,
  modelCatalog,
  modelFigures,
  modelInfo,
}) {
  const { t } = useTranslation();
  const localizedMetricsChartData = metricsChartData.map((item) => ({
    ...item,
    name: metricChartLabel(t, item.name),
  }));

  return (
    <>
      <section className="grid-layout">
        <div className="panel">
          <h2>{t("model.trained")}</h2>

          {modelInfo ? (
            <div className="metric-grid">
              <div>
                <span>{t("common.model")}</span>
                <strong>{modelInfo.model_name}</strong>
              </div>
              <div>
                <span>{t("model.features")}</span>
                <strong>{modelInfo.feature_mode}</strong>
              </div>
              <div>
                <span>{t("model.frequency")}</span>
                <strong>{modelInfo.sfreq} Hz</strong>
              </div>
              <div>
                <span>{t("model.epochSize")}</span>
                <strong>{modelInfo.epoch_size}</strong>
              </div>
              <div>
                <span>{t("model.epochStep")}</span>
                <strong>{modelInfo.step_size}</strong>
              </div>
              <div>
                <span>{t("model.featureCount")}</span>
                <strong>{modelInfo.n_features ?? "N/A"}</strong>
              </div>
            </div>
          ) : (
            <p>{t("model.loadingInfo")}</p>
          )}
        </div>

        <div className="panel">
          <h2>{t("model.cvMetrics")}</h2>

          {metrics ? (
            <>
              <div className="chart-box">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={localizedMetricsChartData}>
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
                  <span>{t("metrics.accuracy")}</span>
                  <strong>{formatMetric(metrics.accuracy_epoch_mean)}</strong>
                </div>
                <div>
                  <span>{t("metrics.balancedAccuracy")}</span>
                  <strong>{formatMetric(metrics.balanced_accuracy_epoch_mean)}</strong>
                </div>
                <div>
                  <span>{t("metrics.precision")}</span>
                  <strong>{formatMetric(metrics.precision_epoch_mean)}</strong>
                </div>
                <div>
                  <span>{t("metrics.recall")}</span>
                  <strong>{formatMetric(metrics.recall_epoch_mean)}</strong>
                </div>
                <div>
                  <span>{t("metrics.f1")}</span>
                  <strong>{formatMetric(metrics.f1_epoch_mean)}</strong>
                </div>
              </div>
            </>
          ) : (
            <p>{t("model.noMetrics")}</p>
          )}
        </div>
      </section>

      <div className="panel model-catalog-panel">
        <h2>{t("model.catalogTitle")}</h2>
        <p className="muted">{t("model.catalogDescription")}</p>

        <ModelCatalogGroup
          models={modelCatalog?.machine_learning}
          t={t}
          title={t("model.mlModels")}
        />
        <ModelCatalogGroup
          models={modelCatalog?.deep_learning}
          t={t}
          title={t("model.dlModels")}
        />
      </div>

      {modelFigures.length > 0 && (
        <div className="panel figures-panel">
          <h2>{t("model.figures")}</h2>

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

ModelView.propTypes = {
  metrics: PropTypes.shape({
    accuracy_epoch_mean: PropTypes.number,
    balanced_accuracy_epoch_mean: PropTypes.number,
    f1_epoch_mean: PropTypes.number,
    precision_epoch_mean: PropTypes.number,
    recall_epoch_mean: PropTypes.number,
  }),
  metricsChartData: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      value: PropTypes.number.isRequired,
    }),
  ).isRequired,
  modelCatalog: PropTypes.shape({
    deep_learning: PropTypes.arrayOf(catalogModelShape),
    machine_learning: PropTypes.arrayOf(catalogModelShape),
  }),
  modelFigures: PropTypes.arrayOf(
    PropTypes.shape({
      title: PropTypes.string.isRequired,
      url: PropTypes.string.isRequired,
    }),
  ).isRequired,
  modelInfo: modelInfoShape,
};

