import { useTranslation } from "react-i18next";

import type { CvMetrics, ModelFigure, ModelInfo } from "./types";
import { formatMetric } from "../../shared/utils/formatters";

interface ModelViewProps {
  metrics: CvMetrics | null;
  modelFigures: ModelFigure[];
  modelInfo: ModelInfo | null;
}
const CV_METRICS = [
  { accent: false, field: "accuracy_epoch_mean", labelKey: "metrics.accuracy" },
  {
    accent: true,
    field: "balanced_accuracy_epoch_mean",
    labelKey: "metrics.balancedAccuracy",
  },
  { accent: false, field: "precision_epoch_mean", labelKey: "metrics.precision" },
  { accent: false, field: "recall_epoch_mean", labelKey: "metrics.recall" },
  { accent: false, field: "f1_epoch_mean", labelKey: "metrics.f1" },
] as const;

export function ModelView({
  metrics,
  modelFigures,
  modelInfo,
}: ModelViewProps) {
  const { t } = useTranslation();

  return (
    <>
      <section className="panel">
        {modelInfo ? (
          <div className="metric-grid model-spec-grid">
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
              <strong>
                {modelInfo.sfreq}
                <span className="unit"> Hz</span>
              </strong>
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
              <strong>{modelInfo.n_features ?? t("common.notAvailable")}</strong>
            </div>
          </div>
        ) : (
          <p className="muted">{t("model.loadingInfo")}</p>
        )}
      </section>

      <section className="panel">
        <span className="eyebrow">{t("model.cvMetrics")}</span>
        <p className="muted cv-metrics-lede">{t("model.cvMetricsDescription")}</p>

        {metrics ? (
          <div className="cv-metrics">
            {CV_METRICS.map((metric) => {
              const value = metrics[metric.field];

              return (
                <div className="cv-metric" key={metric.field}>
                  <div className="cv-metric-head">
                    <span>{t(metric.labelKey)}</span>
                    <span className={metric.accent ? "accent" : ""}>
                      {formatMetric(value)}
                    </span>
                  </div>
                  <span className="cv-metric-track">
                    <span
                      className={metric.accent ? "accent" : ""}
                      style={{ width: `${(value ?? 0) * 100}%` }}
                    />
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">{t("model.noMetrics")}</p>
        )}
      </section>

      {modelFigures.length > 0 && (
        <section className="panel">
          <span className="eyebrow">{t("model.figures")}</span>

          <div className="figures-grid">
            {modelFigures.map((figure) => (
              <figure key={figure.url} className="model-figure">
                <img src={figure.url} alt={figure.title} />
                <figcaption>{figure.title}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
