import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import { AppRouter } from "./app/AppRouter";
import { AuthProvider } from "./features/auth/AuthContext";
import "./i18n";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("No se encontro el elemento raiz de la aplicacion");
}

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <AppRouter />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
