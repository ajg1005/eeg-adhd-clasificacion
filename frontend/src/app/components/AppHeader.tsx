import { useTranslation } from "react-i18next";

import { Tabs } from "../../shared/components/Tabs";
import { TAB_GROUPS } from "../tabs";
import type { TabId } from "../tabs";

interface AppHeaderProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function AppHeader({ activeTab, onTabChange }: AppHeaderProps) {
  const { i18n, t } = useTranslation();

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <span className="brand-name">{t("app.brand")}</span>

        <Tabs
          activeTab={activeTab}
          onTabChange={onTabChange}
          tabGroups={TAB_GROUPS}
        />

        <label className="language-selector">
          <span className="sr-only">{t("app.language")}</span>
          <select
            onChange={(event) => {
              void i18n.changeLanguage(event.target.value);
            }}
            value={i18n.resolvedLanguage || "es"}
          >
            <option value="es">ES</option>
            <option value="en">EN</option>
          </select>
        </label>
      </div>
    </header>
  );
}
