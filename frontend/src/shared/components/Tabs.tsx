import { Fragment } from "react";
import { useTranslation } from "react-i18next";

import type { TabGroup, TabId } from "../../app/tabs";

interface TabsProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  tabGroups: readonly TabGroup[];
}
const NUMBERED_GROUP_ID = "trainingFlow";
export function Tabs({ activeTab, onTabChange, tabGroups }: TabsProps) {
  const { t } = useTranslation();

  return (
    <nav className="tabs">
      {tabGroups.map((group, groupIndex) => (
        <Fragment key={group.id}>
          {groupIndex > 0 && (
            <span aria-hidden="true" className="tab-separator" />
          )}
          <div
            aria-label={t(group.labelKey)}
            className="tab-group"
            role="group"
          >
            {group.tabs.map((tab, index) => (
              <button
                aria-current={activeTab === tab ? "page" : undefined}
                className={
                  activeTab === tab ? "tab-button active" : "tab-button"
                }
                key={tab}
                onClick={() => {
                  onTabChange(tab);
                }}
                type="button"
              >
                {group.id === NUMBERED_GROUP_ID && (
                  <span aria-hidden="true" className="tab-index">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                )}
                {t(`tabs.${tab}`)}
              </button>
            ))}
          </div>
        </Fragment>
      ))}
    </nav>
  );
}
