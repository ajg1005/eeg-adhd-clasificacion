import type { FormEventHandler, ReactNode } from "react";
import { useTranslation } from "react-i18next";

interface AuthFormProps {
  children?: ReactNode;
  email: string;
  error: string | null;
  isSubmitting: boolean;
  onEmailChange: (email: string) => void;
  onPasswordChange: (password: string) => void;
  onSubmit: FormEventHandler<HTMLFormElement>;
  password: string;
  passwordAutoComplete: "current-password" | "new-password";
  submitLabel: string;
  submittingLabel: string;
}

export function AuthForm({
  children,
  email,
  error,
  isSubmitting,
  onEmailChange,
  onPasswordChange,
  onSubmit,
  password,
  passwordAutoComplete,
  submitLabel,
  submittingLabel,
}: AuthFormProps) {
  const { t } = useTranslation();

  return (
    <form className="auth-form" onSubmit={onSubmit}>
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
            onEmailChange(event.target.value);
          }}
          required
          type="email"
          value={email}
        />
      </label>

      <label>
        <span>{t("auth.fields.password")}</span>
        <input
          autoComplete={passwordAutoComplete}
          minLength={8}
          onChange={(event) => {
            onPasswordChange(event.target.value);
          }}
          required
          type="password"
          value={password}
        />
      </label>

      {children}

      <button
        className="primary-button auth-submit"
        disabled={isSubmitting}
        type="submit"
      >
        {isSubmitting ? submittingLabel : submitLabel}
      </button>
    </form>
  );
}
