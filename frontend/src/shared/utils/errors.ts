import i18n from "../../i18n";

export function translate(key: string): string {
  return i18n.t(key);
}
export function errorMessage(error: unknown, fallbackKey: string): string {
  return error instanceof Error ? error.message : translate(fallbackKey);
}
