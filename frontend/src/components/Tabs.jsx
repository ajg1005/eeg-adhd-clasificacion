import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

export function Tabs({ activeTab, disabled = false, onTabChange, tabGroups }) {
  const { t } = useTranslation();

  return (
    <nav className="tabs">
      {tabGroups.map((group) => (
        <div className="tab-group" key={group.id}>
          <span className="tab-group-label">{t(group.labelKey)}</span>
          <div className="tab-group-buttons">
            {group.tabs.map((tab) => (
              <button
                className={activeTab === tab ? "tab-button active" : "tab-button"}
                disabled={disabled && activeTab !== tab}
                key={tab}
                onClick={() => onTabChange(tab)}
                type="button"
              >
                {t(`tabs.${tab}`)}
              </button>
            ))}
          </div>
        </div>
      ))}
    </nav>
  );
}

Tabs.propTypes = {
  activeTab: PropTypes.string.isRequired,
  disabled: PropTypes.bool,
  onTabChange: PropTypes.func.isRequired,
  tabGroups: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelKey: PropTypes.string.isRequired,
      tabs: PropTypes.arrayOf(PropTypes.string).isRequired,
    }),
  ).isRequired,
};
