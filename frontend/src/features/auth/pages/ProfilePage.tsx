import { useTranslation } from "react-i18next";
import { Link, Navigate } from "react-router";

import { LanguageSelector } from "../../../app/components/LanguageSelector";
import { useAuth } from "../useAuth";

export function ProfilePage() {
  const { i18n, t } = useTranslation();
  const { logout, user } = useAuth();

  if (!user) {
    return <Navigate replace to="/login" />;
  }

  const createdAt = new Intl.DateTimeFormat(i18n.resolvedLanguage || "es", {
    dateStyle: "long",
    timeStyle: "short",
  }).format(new Date(user.created_at));

  return (
    <div className="profile-page">
      <header className="account-header">
        <Link className="brand-name brand-link" to="/app">
          {t("app.brand")}
        </Link>
        <div className="account-header-actions">
          <LanguageSelector />
          <Link className="text-link" to="/app">
            {t("auth.profile.back")}
          </Link>
          <button
            className="text-button"
            onClick={logout}
            type="button"
          >
            {t("auth.logout")}
          </button>
        </div>
      </header>

      <main className="profile-shell">
        <div className="view-heading">
          <p className="eyebrow">{t("auth.profile.eyebrow")}</p>
          <h1>{t("auth.profile.title")}</h1>
          <p className="view-lede">{t("auth.profile.description")}</p>
        </div>

        <section aria-label={t("auth.profile.details")} className="profile-details">
          <div>
            <span>{t("auth.fields.email")}</span>
            <strong>{user.email}</strong>
          </div>
          <div>
            <span>{t("auth.profile.status")}</span>
            <strong>
              {user.is_active
                ? t("auth.profile.active")
                : t("auth.profile.inactive")}
            </strong>
          </div>
          <div>
            <span>{t("auth.profile.createdAt")}</span>
            <strong>{createdAt}</strong>
          </div>
          <div>
            <span>{t("auth.profile.identifier")}</span>
            <strong>#{user.id}</strong>
          </div>
        </section>
      </main>
    </div>
  );
}