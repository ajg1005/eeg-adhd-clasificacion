import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

export function AppHeader({ apiStatus }) {
  const { i18n, t } = useTranslation();

  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">EEG ADHD Classifier</p>
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
