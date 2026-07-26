import type { ChangeEvent } from "react";
import { useTranslation } from "react-i18next";

import type {
  ModelInfo,
  PredictionResult,
  ValidationResult,
} from "./types";
import { formatPercent } from "../../shared/utils/formatters";

interface PredictionViewProps {
  decisionScore: number | null;
  file: File | null;
  loadingPrediction: boolean;
  loadingValidation: boolean;
  modelAvailable: boolean;
  modelInfo: ModelInfo | null;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  onPredict: () => Promise<void>;
  prediction: PredictionResult | null;
  validation: ValidationResult | null;
}

interface PredictionDistributionProps {
  prediction: PredictionResult;
}

function classWindowCount(
  prediction: PredictionResult,
  label: string,
): number {
  return prediction.epoch_count_by_class[label] ?? 0;
}

function classWindowPercentage(
  prediction: PredictionResult,
  label: string,
): number {
  if (!prediction.n_epochs) {
    return 0;
  }

  return (classWindowCount(prediction, label) / prediction.n_epochs) * 100;
}

function PredictionDistribution({
  prediction,
}: PredictionDistributionProps) {
  const { t } = useTranslation();
  const classes = [
    {
      className: "control",
      count: classWindowCount(prediction, "Control"),
      label: t("prediction.controlLabel"),
      percentage: classWindowPercentage(prediction, "Control"),
    },
    {
      className: "adhd",
      count: classWindowCount(prediction, "ADHD"),
      label: t("prediction.adhdLabel"),
      percentage: classWindowPercentage(prediction, "ADHD"),
    },
  ];

  return (
    <div className="prediction-distribution">
      {/* El reparto ya no necesita explicacion: el veredicto de arriba dice que
          la clase sale por voto mayoritario y sobre cuantas ventanas. */}
      <span className="eyebrow">{t("prediction.distributionTitle")}</span>

      <div
        aria-label={t("prediction.distributionTitle")}
        className="distribution-bar"
        role="img"
      >
        {classes.map((item) => (
          <span
            className={`distribution-segment ${item.className}`}
            key={item.className}
            style={{ width: `${item.percentage}%` }}
            title={`${item.label}: ${formatPercent(item.percentage / 100)}`}
          />
        ))}
      </div>

      <div className="distribution-legend distribution-legend-inline">
        {classes.map((item) => (
          <div className="distribution-legend-row" key={item.className}>
            <span className={`legend-dot ${item.className}`} />
            <span className="distribution-label">{item.label}</span>
            <strong>
              {item.count}/{prediction.n_epochs}
            </strong>
            <span>{formatPercent(item.percentage / 100)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PredictionView({
  decisionScore,
  file,
  loadingPrediction,
  loadingValidation,
  modelAvailable,
  modelInfo,
  onFileChange,
  onPredict,
  prediction,
  validation,
}: PredictionViewProps) {
  const { t } = useTranslation();

  return (
    <section className="grid-layout">
      <div className="panel">
        <h2>{t("prediction.patientFile")}</h2>

        <label className="file-drop">
          <span>{t("prediction.selectCsv")}</span>
          <input
            type="file"
            accept=".csv"
            onChange={(event) => {
              void onFileChange(event);
            }}
          />
        </label>

        {file && (
          <div className="file-info">
            <strong>{file.name}</strong>
            <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>
        )}

        {loadingValidation && (
          <div className="alert alert-info">{t("prediction.validating")}</div>
        )}

        {validation && (
          <div className="alert alert-success">
            {t("prediction.validCsv", {
              channels: validation.available_channels.length,
              rows: validation.rows,
            })}
          </div>
        )}

        {modelInfo && validation && (
          <div className="channel-validation">
            <p className="muted">{t("prediction.expectedChannels")}</p>
            <div className="channel-list">
              {modelInfo.channels.map((channel) => (
                <span
                  className={
                    validation.available_channels.includes(channel)
                      ? "channel-ok"
                      : ""
                  }
                  key={channel}
                >
                  {channel}
                </span>
              ))}
            </div>
          </div>
        )}

        <button
          className="primary-button"
          disabled={
            !modelAvailable ||
            !file ||
            loadingPrediction ||
            loadingValidation
          }
          onClick={() => {
            void onPredict();
          }}
          type="button"
        >
          {loadingPrediction ? t("prediction.processing") : t("prediction.run")}
        </button>

        {loadingPrediction && (
          <p className="muted">{t("prediction.processingHint")}</p>
        )}
      </div>

      <div className="panel">
        <h2>{t("prediction.result")}</h2>

        {prediction ? (
          <>
            {/* El veredicto es el dato de la pantalla: manda en tamaño, y la
                confianza media lo acompaña como unico acento en cobre. */}
            <div className="prediction-verdict">
              <div>
                <h2 className="prediction-label">
                  {prediction.prediction_label}
                </h2>
                <p className="muted">
                  {t("prediction.majorityVote", {
                    count: classWindowCount(
                      prediction,
                      prediction.prediction_label,
                    ),
                    total: prediction.n_epochs,
                  })}
                </p>
              </div>
              <div className="prediction-confidence">
                <span className="eyebrow">{t("prediction.meanConfidence")}</span>
                <strong>{formatPercent(decisionScore)}</strong>
              </div>
            </div>

            <PredictionDistribution prediction={prediction} />

            <p className="muted prediction-summary">
              {t("prediction.summary", {
                epochs: prediction.n_epochs,
                model: prediction.model_name || modelInfo?.model_name,
              })}
            </p>
          </>
        ) : (
          <p className="muted">{t("prediction.empty")}</p>
        )}

        {/* Fuera del condicional: el aviso vale tanto antes como despues de
            predecir, y no es algo que deba aparecer solo a posteriori. */}
        <p className="prediction-disclaimer">{t("prediction.disclaimer")}</p>
      </div>
    </section>
  );
}
