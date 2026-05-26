# Análisis de licencias de dependencias

Este documento recopila la licencia de cada dependencia directa utilizada en el
proyecto **EEG ADHD Classifier**, junto con su compatibilidad respecto a la
licencia elegida para el proyecto (MIT).

## Licencia del proyecto

El proyecto se distribuye bajo **MIT License** (ver fichero `LICENSE` en la raíz
del repositorio). MIT es una licencia permisiva que permite uso, modificación y
redistribución (incluso comercial) siempre que se mantenga el aviso de copyright.

## Dependencias

### Backend — Producción (`requirements.txt`)

| Librería          | Versión  | Licencia                            | Compatible con MIT |
|-------------------|----------|-------------------------------------|--------------------|
| numpy             | 2.1.3    | BSD-3-Clause                        | Sí                 |
| pandas            | 2.2.3    | BSD-3-Clause                        | Sí                 |
| scikit-learn      | 1.8.0    | BSD-3-Clause                        | Sí                 |
| scipy             | 1.14.1   | BSD-3-Clause                        | Sí                 |
| matplotlib        | 3.9.2    | Matplotlib License (BSD-style, PSF) | Sí                 |
| joblib            | 1.4.2    | BSD-3-Clause                        | Sí                 |
| fastapi           | 0.115.6  | MIT                                 | Sí                 |
| uvicorn           | 0.32.1   | BSD-3-Clause                        | Sí                 |
| python-multipart  | 0.0.20   | Apache 2.0                          | Sí                 |
| xgboost           | 2.1.3    | Apache 2.0                          | Sí                 |
| tensorflow        | 2.21.0   | Apache 2.0                          | Sí                 |

### Backend — Desarrollo (`requirements-dev.txt`)

| Librería     | Versión  | Licencia      | Compatible con MIT |
|--------------|----------|---------------|--------------------|
| pytest       | 8.3.5    | MIT           | Sí                 |
| pytest-cov   | 6.1.1    | MIT           | Sí                 |
| ruff         | 0.11.8   | MIT           | Sí                 |
| httpx        | 0.28.1   | BSD-3-Clause  | Sí                 |

### Frontend — Producción (`package.json` → `dependencies`)

| Librería   | Versión | Licencia | Compatible con MIT |
|------------|---------|----------|--------------------|
| react      | 19.2.5  | MIT      | Sí                 |
| react-dom  | 19.2.5  | MIT      | Sí                 |
| recharts   | 3.8.1   | MIT      | Sí                 |

### Frontend — Desarrollo (`package.json` → `devDependencies`)

| Librería                       | Versión  | Licencia | Compatible con MIT |
|--------------------------------|----------|----------|--------------------|
| @eslint/js                     | 10.0.1   | MIT      | Sí                 |
| @types/react                   | 19.2.14  | MIT      | Sí                 |
| @types/react-dom               | 19.2.3   | MIT      | Sí                 |
| @vitejs/plugin-react           | 6.0.1    | MIT      | Sí                 |
| eslint                         | 10.2.1   | MIT      | Sí                 |
| eslint-plugin-react-hooks      | 7.1.1    | MIT      | Sí                 |
| eslint-plugin-react-refresh    | 0.5.2    | MIT      | Sí                 |
| globals                        | 17.5.0   | MIT      | Sí                 |
| vite                           | 8.0.10   | MIT      | Sí                 |

## Resumen por familia de licencia

| Familia de licencia | Nº de dependencias | Compatibilidad con MIT |
|---------------------|--------------------|------------------------|
| MIT                 | 17                 | Total                  |
| BSD-3-Clause        | 7                  | Total                  |
| Apache 2.0          | 3                  | Total                  |
| **TOTAL**           | **27**             | **100% compatibles**   |

## Conclusiones

El proyecto se distribuye bajo licencia MIT. Se ha verificado la compatibilidad
de las 27 dependencias utilizadas (11 de backend en producción, 4 de
desarrollo, 3 de frontend en producción y 9 de desarrollo), comprobando que
todas ellas se distribuyen bajo licencias permisivas (MIT, BSD-3-Clause o
Apache 2.0). Ninguna de las dependencias está sujeta a licencias copyleft
(como GPL o AGPL), por lo que no existen restricciones que obliguen a
relicenciar el código fuente del proyecto.

La elección de MIT como licencia se justifica por los siguientes motivos:

1. Es una licencia permisiva ampliamente reconocida en la comunidad open source.
2. Facilita la reutilización académica y profesional del código.
3. Es compatible con todas las dependencias seleccionadas.

## Notas

- Esta tabla recoge únicamente las **dependencias directas** declaradas en
  `requirements.txt`, `requirements-dev.txt` y `frontend/package.json`. No se
  incluyen dependencias transitivas (paquetes instalados como dependencia de
  otra dependencia).
- Para una auditoría exhaustiva incluyendo dependencias transitivas pueden
  utilizarse herramientas automatizadas como `pip-licenses` (Python) o
  `license-checker` (npm).
