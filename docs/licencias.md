# Análisis de licencias de dependencias

Este documento recopila la licencia de cada dependencia directa utilizada en el
proyecto **EEG ADHD Classifier**, junto con su compatibilidad respecto a la
licencia elegida para el proyecto (MIT).

## Licencia del proyecto

El proyecto se distribuye bajo **MIT License** (ver fichero `LICENSE` en la raíz
del repositorio). MIT es una licencia permisiva que permite uso, modificación y
redistribución (incluso comercial) siempre que se mantenga el aviso de copyright.

## Dependencias

### Backend — Producción (`pyproject.toml` y `uv.lock`)

| Librería          | Versión  | Licencia                            | Compatible con MIT |
|-------------------|----------|-------------------------------------|--------------------|
| numpy             | 2.5.1    | BSD-3-Clause                        | Sí                 |
| pandas            | 3.0.3    | BSD-3-Clause                        | Sí                 |
| scikit-learn      | 1.8.0    | BSD-3-Clause                        | Sí                 |
| scipy             | 1.18.0   | BSD-3-Clause                        | Sí                 |
| matplotlib        | 3.11.1   | Matplotlib License (BSD-style, PSF) | Sí                 |
| joblib            | 1.5.3    | BSD-3-Clause                        | Sí                 |
| fastapi           | 0.139.2  | MIT                                 | Sí                 |
| uvicorn           | 0.51.0   | BSD-3-Clause                        | Sí                 |
| python-multipart  | 0.0.32   | Apache 2.0                          | Sí                 |
| xgboost           | 3.3.0    | Apache 2.0                          | Sí                 |
| tensorflow        | 2.21.0   | Apache 2.0                          | Sí                 |
| SQLAlchemy        | 2.0.51   | MIT                                 | Sí                 |
| alembic           | 1.18.5   | MIT                                 | Sí                 |
| psycopg[binary]   | 3.3.4    | LGPL-3.0-or-later                   | Sí (ver nota)      |
| celery[redis]     | 5.6.3    | BSD-3-Clause                        | Sí                 |

> **Nota sobre psycopg.** psycopg 3 se distribuye bajo licencia
> **LGPL-3.0-or-later**, una licencia *copyleft débil*. Su uso como librería
> importada por una aplicación Python (linking dinámico) es compatible con
> licencias permisivas como MIT, siempre que el usuario final pueda sustituir
> la versión de psycopg utilizada. En este proyecto la librería se importa
> como dependencia externa a través de `uv` sin que se modifique su código
> fuente, por lo que **no se aplica el efecto copyleft** sobre el código
> propio del TFG. El requisito práctico es mencionar el uso de psycopg y su
> licencia en la documentación, cosa que se hace en esta tabla.

### Backend — Desarrollo (`pyproject.toml`, grupo `dev`)

| Librería     | Versión  | Licencia      | Compatible con MIT |
|--------------|----------|---------------|--------------------|
| pytest       | 9.1.1    | MIT           | Sí                 |
| pytest-cov   | 7.1.0    | MIT           | Sí                 |
| ruff         | 0.15.22  | MIT           | Sí                 |
| httpx2       | 2.7.0    | BSD-3-Clause  | Sí                 |

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

| Familia de licencia   | Nº de dependencias | Compatibilidad con MIT  |
|-----------------------|--------------------|-------------------------|
| MIT                   | 18                 | Total                   |
| BSD/BSD-style         | 9                  | Total                   |
| Apache 2.0            | 3                  | Total                   |
| LGPL-3.0-or-later     | 1                  | Total (uso como lib)    |
| **TOTAL**             | **31**             | **100% compatibles**    |

## Conclusiones

El proyecto se distribuye bajo licencia MIT. Se ha verificado la compatibilidad
de las 31 dependencias directas utilizadas (15 de backend en producción, 4 de
desarrollo, 3 de frontend en producción y 9 de desarrollo), comprobando que
todas ellas son compatibles con la licencia MIT del proyecto.

La distribución por familias es:

- **MIT, BSD/BSD-style y Apache 2.0** (30 dependencias): licencias permisivas
  totalmente compatibles con MIT, sin restricciones adicionales más allá de
  mantener el aviso de copyright original.
- **LGPL-3.0-or-later** (1 dependencia, psycopg): licencia *copyleft débil*
  compatible cuando la librería se utiliza importada sin modificarla, como
  es el caso de este proyecto.

Ninguna dependencia exige relicenciar el código propio del proyecto bajo una
licencia copyleft fuerte (GPL, AGPL).

La elección de MIT como licencia del TFG se justifica por:

1. Es una licencia permisiva ampliamente reconocida en la comunidad open source.
2. Facilita la reutilización académica y profesional del código.
3. Es compatible con todas las dependencias seleccionadas.
4. Es una **licencia de software** (la guía oficial de TFG de la UBU exige
   que la licencia del proyecto sea de software, excluyendo explícitamente
   Creative Commons).

## Notas

- Esta tabla recoge únicamente las **dependencias directas** declaradas en
  `pyproject.toml` y `frontend/package.json`. Las versiones concretas de Python
  corresponden a `uv.lock`. No
  se incluyen dependencias transitivas (paquetes instalados como dependencia
  de otra dependencia).
- Para una auditoría exhaustiva incluyendo dependencias transitivas pueden
  utilizarse herramientas automatizadas como `pip-licenses` (Python) o
  `license-checker` (npm).
