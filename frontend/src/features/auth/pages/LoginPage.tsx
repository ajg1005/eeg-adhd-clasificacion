import { useState } from "react";
import type { FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation, useNavigate } from "react-router";

import { errorMessage } from "../../../shared/utils/errors";
import { AuthForm } from "../components/AuthForm";
import { AuthLayout } from "../components/AuthLayout";
import { useAuth } from "../useAuth";

interface LoginLocationState {
  from?: string;
}

export function LoginPage() {
  const { t } = useTranslation();
  const { login } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const locationState = location.state as LoginLocationState | null;
  const destination =
    locationState?.from === "/profile" ? "/profile" : "/app";

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      await login({ email, password });
      void navigate(destination, { replace: true });
    } catch (caughtError) {
      setError(errorMessage(caughtError, "errors.auth.login"));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout
      description={t("auth.login.description")}
      footer={
        <p>
          {t("auth.login.noAccount")}{" "}
          <Link to="/register">{t("auth.login.createAccount")}</Link>
        </p>
      }
      title={t("auth.login.title")}
    >
      <AuthForm
        email={email}
        error={error}
        isSubmitting={isSubmitting}
        onEmailChange={setEmail}
        onPasswordChange={setPassword}
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
        password={password}
        passwordAutoComplete="current-password"
        submitLabel={t("auth.login.submit")}
        submittingLabel={t("auth.login.submitting")}
      />
    </AuthLayout>
  );
}