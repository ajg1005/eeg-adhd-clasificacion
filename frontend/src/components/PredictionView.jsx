import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import { fileShape, modelInfoShape } from "../propTypes";
import { formatPercent } from "../utils/formatters";

function classWindowCount(prediction, label) {
  return prediction?.epoch_count_by_class?.[label] || 0;
}

function classWindowPercentage(prediction, label) {
  if (!prediction?.n_epochs) {
    return 0;
  }

  return (classWindowCount(prediction, label) / prediction.n_epochs) * 100;
}

function PredictionDistribution({ prediction }) {
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
      <h3>{t("prediction.distributionTitle")}</h3>
      <p className="muted">{t("prediction.distributionDescription")}</p>

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

      <div className="distribution-legend">
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
  modelInfo,
  onFileChange,
  onPredict,
  prediction,
  validation,
}) {
  const { t } = useTranslation();

  return (
    <section className="grid-layout">
      <div className="panel">
        <h2>{t("prediction.patientFile")}</h2>

        <label className="file-drop">
          <span>{t("prediction.selectCsv")}</span>
          <input type="file" accept=".csv" onChange={onFileChange} />
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
          </div>
        )}

        <button
          className="primary-button"
          disabled={!file || loadingPrediction || loadingValidation}
          onClick={onPredict}
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
            <div className="result-main">
              <div>
                <span>{t("prediction.classification")}</span>
                <strong>{prediction.prediction_label}</strong>
              </div>
              <div>
                <span>{t("prediction.meanConfidence")}</span>
                <strong>{formatPercent(decisionScore)}</strong>
              </div>
              <div>
                <span>{t("prediction.controlWindows")}</span>
                <strong>
                  {classWindowCount(prediction, "Control")}/{prediction.n_epochs}
                </strong>
              </div>
              <div>
                <span>{t("prediction.adhdWindows")}</span>
                <strong>
                  {classWindowCount(prediction, "ADHD")}/{prediction.n_epochs}
                </strong>
              </div>
            </div>

            <p className="muted">
              {t("prediction.summary", {
                epochs: prediction.n_epochs,
                model: prediction.model_name || modelInfo?.model_name,
              })}
            </p>

            <PredictionDistribution prediction={prediction} />
          </>
        ) : (
          <p className="muted">{t("prediction.empty")}</p>
        )}
      </div>
    </section>
  );
}

PredictionView.propTypes = {
  decisionScore: PropTypes.number,
  file: fileShape,
  loadingPrediction: PropTypes.bool.isRequired,
  loadingValidation: PropTypes.bool.isRequired,
  modelInfo: modelInfoShape,
  onFileChange: PropTypes.func.isRequired,
  onPredict: PropTypes.func.isRequired,
  prediction: PropTypes.shape({
    epoch_count_by_class: PropTypes.objectOf(PropTypes.number),
    model_name: PropTypes.string,
    n_epochs: PropTypes.number.isRequired,
    prediction_label: PropTypes.string.isRequired,
  }),
  validation: PropTypes.shape({
    available_channels: PropTypes.arrayOf(PropTypes.string).isRequired,
    rows: PropTypes.number.isRequired,
  }),
};

PredictionDistribution.propTypes = {
  prediction: PropTypes.shape({
    epoch_count_by_class: PropTypes.objectOf(PropTypes.number),
    n_epochs: PropTypes.number.isRequired,
  }).isRequired,
};
