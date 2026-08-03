import { useTranslation } from "react-i18next";

export function LanguageSelector() {
  const { i18n, t } = useTranslation();

  return (
    <label className="language-selector">
      <span className="sr-only">{t("app.language")}</span>
      <select
        aria-label={t("app.language")}
        onChange={(event) => {
          void i18n.changeLanguage(event.target.value);
        }}
        value={i18n.resolvedLanguage || "es"}
      >
        <option value="es">ES</option>
        <option value="en">EN</option>
      </select>
    </label>
  );
}