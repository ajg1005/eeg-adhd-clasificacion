import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function SignalView({
  file,
  loadingPreview,
  maxPoints,
  modelInfo,
  onLoadPreview,
  selectedChannel,
  setMaxPoints,
  setSelectedChannel,
  signalPreview,
}) {
  return (
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
          onClick={onLoadPreview}
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
                stroke="#be5a38"
                strokeWidth={2}
                type="monotone"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

