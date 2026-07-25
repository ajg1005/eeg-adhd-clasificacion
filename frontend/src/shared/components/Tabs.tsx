import { useTranslation } from "react-i18next";

import type { TabGroup, TabId } from "../../app/tabs";

interface TabsProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  tabGroups: readonly TabGroup[];
}

export function Tabs({ activeTab, onTabChange, tabGroups }: TabsProps) {
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
                key={tab}
                onClick={() => {
                  onTabChange(tab);
                }}
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
