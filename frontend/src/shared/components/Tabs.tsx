import { Fragment } from "react";
import { useTranslation } from "react-i18next";

import type { TabGroup, TabId } from "../../app/tabs";

interface TabsProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  tabGroups: readonly TabGroup[];
}

// La navegacion vive dentro de la barra fija de 64px, donde no cabe la etiqueta
// de cada grupo: visualmente se separan con una regla, y el texto del grupo pasa
// a ser el nombre accesible para que no se pierda la agrupacion.
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
            {group.tabs.map((tab) => (
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
                {t(`tabs.${tab}`)}
              </button>
            ))}
          </div>
        </Fragment>
      ))}
    </nav>
  );
}
