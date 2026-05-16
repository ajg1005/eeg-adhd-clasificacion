export function DataView({
  classFilter,
  datasetSummary,
  file,
  loadingDatasetSummary,
  loadingValidation,
  maxPatients,
  modelInfo,
  onClassFilterChange,
  onFileChange,
  onMaxPatientsChange,
  validation,
}) {
  return (
    <section className="data-layout">
      <div className="grid-layout">
        <div className="panel">
          <h2>Archivo EEG</h2>

          <label className="file-drop">
            <span>Seleccionar CSV</span>
            <input type="file" accept=".csv" onChange={onFileChange} />
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
      </div>

      <div className="panel dataset-panel">
        <div className="section-heading-row">
          <div>
            <h2>Estadísticas del dataset</h2>
            <p className="muted">
              Resumen general, distribución de clases y selección de pacientes.
            </p>
          </div>

          <div className="controls-row compact-controls">
            <label>
              Clase
              <select
                disabled={!file || loadingDatasetSummary}
                value={classFilter}
                onChange={onClassFilterChange}
              >
                <option value="all">Todos</option>
                <option value="adhd">TDAH</option>
                <option value="control">Control</option>
              </select>
            </label>

            <label>
              Pacientes
              <input
                disabled={!file || loadingDatasetSummary}
                max="100"
                min="1"
                step="1"
                type="number"
                value={maxPatients}
                onChange={onMaxPatientsChange}
              />
            </label>
          </div>
        </div>

        {loadingDatasetSummary && <p>Calculando estadísticas...</p>}

        {!file && (
          <p className="muted">Sube un CSV para ver estadísticas del dataset.</p>
        )}

        {datasetSummary && (
          <>
            <div className="metric-grid dataset-summary-grid">
              <div>
                <span>Pacientes totales</span>
                <strong>{datasetSummary.total_patients || "N/A"}</strong>
              </div>
              <div>
                <span>Pacientes filtrados</span>
                <strong>{datasetSummary.filtered_patients_count || "N/A"}</strong>
              </div>
              <div>
                <span>Canales EEG</span>
                <strong>{datasetSummary.n_eeg_channels}</strong>
              </div>
              <div>
                <span>Mostrando</span>
                <strong>{datasetSummary.shown_patients_count}</strong>
              </div>
            </div>

            <div className="class-counts">
              {Object.entries(datasetSummary.class_counts).map(([label, count]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>

            {datasetSummary.patients.length > 0 ? (
              <div className="patient-table-wrap">
                <table className="patient-table">
                  <thead>
                    <tr>
                      <th>Paciente</th>
                      <th>Clase</th>
                      <th>Muestras</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetSummary.patients.map((patient) => (
                      <tr key={patient.patient_id}>
                        <td>{patient.patient_id}</td>
                        <td>{patient.class_label}</td>
                        <td>{patient.n_samples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">No hay pacientes para el filtro seleccionado.</p>
            )}
          </>
        )}
      </div>
    </section>
  );
}

