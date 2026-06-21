import PropTypes from "prop-types";
import { BrainCircuit } from "lucide-react";
import { useTranslation } from "react-i18next";

export function AppHeader({ apiStatus }) {
  const { i18n, t } = useTranslation();

  return (
    <header className="app-header">
      <div className="header-copy">
        <div className="brand-mark">
          <span className="brand-icon" aria-hidden="true">
            <BrainCircuit size={20} strokeWidth={2.4} />
          </span>
          <p className="eyebrow">EEG ADHD Classifier</p>
        </div>
        <h1>{t("app.title")}</h1>
        <p className="subtitle">{t("app.subtitle")}</p>
      </div>

      <div className="header-actions">
        <label className="language-selector">
          <span>{t("app.language")}</span>
          <select
            onChange={(event) => i18n.changeLanguage(event.target.value)}
            value={i18n.resolvedLanguage || "es"}
          >
            <option value="es">ES</option>
            <option value="en">EN</option>
          </select>
        </label>
        <div className={`api-pill api-pill-${apiStatus}`}>
          {t("app.api")}: {apiStatus === "ok" ? "online" : apiStatus}
        </div>
      </div>
    </header>
  );
}

AppHeader.propTypes = {
  apiStatus: PropTypes.string.isRequired,
};
