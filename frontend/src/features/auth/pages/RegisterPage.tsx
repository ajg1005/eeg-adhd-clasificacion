import { useState } from "react";
import type { FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router";

import { errorMessage } from "../../../shared/utils/errors";
import { AuthLayout } from "../components/AuthLayout";
import { useAuth } from "../useAuth";

export function RegisterPage() {
  const { t } = useTranslation();
  const { register } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirmation, setPasswordConfirmation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (password !== passwordConfirmation) {
      setError(t("auth.register.passwordMismatch"));
      return;
    }

    setIsSubmitting(true);

    try {
      await register({ email, password });
      void navigate("/app", { replace: true });
    } catch (caughtError) {
      setError(errorMessage(caughtError, "errors.auth.register"));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout
      description={t("auth.register.description")}
      footer={
        <p>
          {t("auth.register.hasAccount")}{" "}
          <Link to="/login">{t("auth.register.signIn")}</Link>
        </p>
      }
      title={t("auth.register.title")}
    >
      <form className="auth-form" onSubmit={(event) => {
        void handleSubmit(event);
      }}>
        {error ? (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        ) : null}

        <label>
          <span>{t("auth.fields.email")}</span>
          <input
            autoComplete="email"
            inputMode="email"
            onChange={(event) => {
              setEmail(event.target.value);
            }}
            required
            type="email"
            value={email}
          />
        </label>

        <label>
          <span>{t("auth.fields.password")}</span>
          <input
            autoComplete="new-password"
            minLength={8}
            onChange={(event) => {
              setPassword(event.target.value);
            }}
            required
            type="password"
            value={password}
          />
        </label>

        <label>
          <span>{t("auth.fields.passwordConfirmation")}</span>
          <input
            autoComplete="new-password"
            minLength={8}
            onChange={(event) => {
              setPasswordConfirmation(event.target.value);
            }}
            required
            type="password"
            value={passwordConfirmation}
          />
        </label>

        <button
          className="primary-button auth-submit"
          disabled={isSubmitting}
          type="submit"
        >
          {isSubmitting
            ? t("auth.register.submitting")
            : t("auth.register.submit")}
        </button>
      </form>
    </AuthLayout>
  );
}