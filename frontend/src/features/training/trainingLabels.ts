import type { TFunction } from "i18next";

type OptionValue = string | number | boolean | null | undefined;

function translateKey(t: TFunction, key: string, fallback: string): string {
  return t(key, { defaultValue: fallback });
}

export function signalParamLabel(t: TFunction, name: string): string {
  return translateKey(t, `training.signalParams.${name}`, name);
}

export function modelParamLabel(t: TFunction, name: string): string {
  return translateKey(t, `training.modelParams.${name}`, name);
}

export function trainingParamLabel(t: TFunction, name: string): string {
  return translateKey(t, `training.trainingParams.${name}`, name);
}

export function optionValueLabel(t: TFunction, value: OptionValue): string {
  if (value === null || value === undefined || value === "none") {
    return t("training.values.none");
  }

  return translateKey(t, `training.values.${String(value)}`, String(value));
}

export function evaluationModeLabel(t: TFunction, value: string): string {
  return translateKey(t, `training.evaluationModes.${value}`, value);
}

export function methodLabel(t: TFunction, value: string): string {
  return translateKey(t, `training.methods.${value}`, value);
}

export function reportRowLabel(t: TFunction, name: string): string {
  return translateKey(t, `training.reportRows.${name}`, name);
}

export function reportColumnLabel(t: TFunction, name: string): string {
  return translateKey(t, `training.reportColumns.${name}`, name);
}
export const REPORT_COLUMNS = [
  "precision",
  "recall",
  "f1-score",
  "support",
] as const;
