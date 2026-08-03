import { Navigate, Outlet, Route, Routes, useLocation } from "react-router";
import { useTranslation } from "react-i18next";

import App from "./App";
import { LoginPage } from "../features/auth/pages/LoginPage";
import { RegisterPage } from "../features/auth/pages/RegisterPage";
import { useAuth } from "../features/auth/useAuth";

function SessionLoading() {
  const { t } = useTranslation();

  return (
    <div className="session-loading" role="status">
      <span aria-hidden="true" className="loading-indicator" />
      <span className="sr-only">{t("common.loading")}</span>
    </div>
  );
}

function RequireAuth() {
  const { isLoading, user } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <SessionLoading />;
  }

  if (!user) {
    return (
      <Navigate
        replace
        state={{ from: location.pathname }}
        to="/login"
      />
    );
  }

  return <Outlet />;
}

function PublicOnly() {
  const { isLoading, user } = useAuth();

  if (isLoading) {
    return <SessionLoading />;
  }

  return user ? <Navigate replace to="/app" /> : <Outlet />;
}

function InitialRedirect() {
  const { isLoading, user } = useAuth();

  if (isLoading) {
    return <SessionLoading />;
  }

  return <Navigate replace to={user ? "/app" : "/login"} />;
}

export function AppRouter() {
  return (
    <Routes>
      <Route element={<InitialRedirect />} path="/" />

      <Route element={<PublicOnly />}>
        <Route element={<LoginPage />} path="/login" />
        <Route element={<RegisterPage />} path="/register" />
      </Route>

      <Route element={<RequireAuth />}>
        <Route element={<App />} path="/app" />
      </Route>

      <Route element={<Navigate replace to="/" />} path="*" />
    </Routes>
  );
}