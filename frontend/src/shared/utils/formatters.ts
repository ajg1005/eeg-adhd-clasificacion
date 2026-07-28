export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return value.toFixed(3);
}
