import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatPercent } from "../utils/formatters";

export function PredictionView({
  adhdEpochs,
  controlEpochs,
  decisionScore,
  file,
  finalClassEpochPercentage,
  loadingPrediction,
  onPredict,
  prediction,
  predictionChartData,
  thresholdUsed,
  validation,
}) {
  return (
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
          onClick={onPredict}
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
                  <Bar dataKey="porcentaje" fill="#be7c4d" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <p className="muted">Todavía no hay predicción ejecutada.</p>
        )}
      </div>
    </section>
  );
}

