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

import { fileShape, modelInfoShape } from "../propTypes";
import { formatPercent } from "../utils/formatters";

export function PredictionView({
  decisionScore,
  file,
  loadingPrediction,
  loadingValidation,
  modelInfo,
  onFileChange,
  onPredict,
  prediction,
  predictionChartData,
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

        {loadingValidation && <p>{t("prediction.validating")}</p>}

        {validation && (
          <p className="ok-text">
            {t("prediction.validCsv", {
              channels: validation.available_channels.length,
              rows: validation.rows,
            })}
          </p>
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
                <span>{t("prediction.confidence")}</span>
                <strong>{formatPercent(decisionScore)}</strong>
              </div>
            </div>

            <p className="muted">
              {t("prediction.summary", {
                epochs: prediction.n_epochs,
                model: prediction.model_name || modelInfo?.model_name,
              })}
            </p>

            <h3>{t("prediction.windowDistribution")}</h3>
            <div className="chart-box">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={predictionChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="clase" />
                  <YAxis domain={[0, 100]} unit="%" />
                  <Tooltip />
                  <Bar dataKey="porcentaje" fill="#be7c4d" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
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
    model_name: PropTypes.string,
    n_epochs: PropTypes.number.isRequired,
    prediction_label: PropTypes.string.isRequired,
  }),
  predictionChartData: PropTypes.arrayOf(
    PropTypes.shape({
      clase: PropTypes.string.isRequired,
      porcentaje: PropTypes.number.isRequired,
    }),
  ).isRequired,
  validation: PropTypes.shape({
    available_channels: PropTypes.arrayOf(PropTypes.string).isRequired,
    rows: PropTypes.number.isRequired,
  }),
};
