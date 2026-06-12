function translateKey(t, key, fallback) {
  return t(key, { defaultValue: fallback });
}

export function signalParamLabel(t, name) {
  return translateKey(t, `training.signalParams.${name}`, name);
}

export function modelParamLabel(t, name) {
  return translateKey(t, `training.modelParams.${name}`, name);
}

export function trainingParamLabel(t, name) {
  return translateKey(t, `training.trainingParams.${name}`, name);
}

export function optionValueLabel(t, value) {
  if (value === null || value === undefined || value === "none") {
    return t("training.values.none");
  }

  return translateKey(t, `training.values.${String(value)}`, String(value));
}

export function evaluationModeLabel(t, value) {
  return translateKey(t, `training.evaluationModes.${value}`, value);
}

export function methodLabel(t, value) {
  return translateKey(t, `training.methods.${value}`, value);
}

export function reportRowLabel(t, name) {
  return translateKey(t, `training.reportRows.${name}`, name);
}

export function reportColumnLabel(t, name) {
  return translateKey(t, `training.reportColumns.${name}`, name);
}

// Columnas estandar del classification_report de scikit-learn.
export const REPORT_COLUMNS = ["precision", "recall", "f1-score", "support"];
