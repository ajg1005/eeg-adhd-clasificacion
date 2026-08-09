# EEGScope Landing

Landing page estática del proyecto EEGScope. Es un proyecto Vite independiente
de la aplicación React principal y no realiza llamadas al backend.

## Desarrollo local

```bash
npm install
npm run dev
```

Vite mostrará la URL local disponible, normalmente `http://localhost:5173`.

## Verificación

```bash
npm run build
npm run preview
```

## Cloudflare Pages

Configura el proyecto de Pages conectado al repositorio con estos valores:

| Ajuste | Valor |
|---|---|
| Root directory | `landing` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Production branch | `main` |

El dominio raíz `eegscope.dev` debe apuntar al proyecto de Cloudflare Pages. La
aplicación principal permanece disponible en `https://app.eegscope.dev`.
