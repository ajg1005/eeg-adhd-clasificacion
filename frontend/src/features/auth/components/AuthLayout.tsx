import type { ReactNode } from "react";
import { Link } from "react-router";

import { LanguageSelector } from "../../../app/components/LanguageSelector";

interface AuthLayoutProps {
  children: ReactNode;
  description: string;
  footer: ReactNode;
  title: string;
}

export function AuthLayout({
  children,
  description,
  footer,
  title,
}: AuthLayoutProps) {
  return (
    <div className="auth-page">
      <header className="auth-header">
        <Link className="brand-name brand-link" to="/">
          EEG ADHD
        </Link>
        <LanguageSelector />
      </header>

      <main className="auth-main">
        <section aria-labelledby="auth-title" className="auth-panel">
          <div className="auth-heading">
            <p className="eyebrow">EEG ADHD Classifier</p>
            <h1 id="auth-title">{title}</h1>
            <p>{description}</p>
          </div>

          {children}

          <div className="auth-footer">{footer}</div>
        </section>
      </main>
    </div>
  );
}