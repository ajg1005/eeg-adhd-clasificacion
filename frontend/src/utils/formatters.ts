// Formatear porcentajes para tarjetas y graficas
export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}

// Formatear metricas del modelo a tres decimales
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return value.toFixed(3);
}