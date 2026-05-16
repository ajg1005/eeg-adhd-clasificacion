// Formatear porcentajes para tarjetas y graficas
export function formatPercent(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}

// Formatear metricas del modelo a tres decimales
export function formatMetric(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toFixed(3);
}

