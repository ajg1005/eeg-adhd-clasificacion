import { useTranslation } from "react-i18next";
import { Link } from "react-router";

import { useAuth } from "../../features/auth/useAuth";
import { Tabs } from "../../shared/components/Tabs";
import { TAB_GROUPS } from "../tabs";
import type { TabId } from "../tabs";
import { LanguageSelector } from "./LanguageSelector";

interface AppHeaderProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function AppHeader({ activeTab, onTabChange }: AppHeaderProps) {
  const { t } = useTranslation();
  const { logout, user } = useAuth();

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <Link className="brand-name brand-link" to="/app">
          {t("app.brand")}
        </Link>

        <Tabs
          activeTab={activeTab}
          onTabChange={onTabChange}
          tabGroups={TAB_GROUPS}
        />

        <div className="app-header-actions">
          <LanguageSelector />
          <Link className="account-link" title={user?.email} to="/profile">
            {user?.email}
          </Link>
          <button className="text-button" onClick={logout} type="button">
            {t("auth.logout")}
          </button>
        </div>
      </div>
    </header>
  );
}
