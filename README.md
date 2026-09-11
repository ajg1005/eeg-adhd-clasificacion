# EEGScope

[![CI](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)
[![Licencia](https://img.shields.io/github/license/ajg1005/eeg-adhd-clasificacion)](LICENSE)
[![Última release](https://img.shields.io/github/v/release/ajg1005/eeg-adhd-clasificacion?include_prereleases)](https://github.com/ajg1005/eeg-adhd-clasificacion/releases)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
[![Abrir EEGScope](https://img.shields.io/badge/Abrir-EEGScope-2563EB)](https://app.eegscope.dev)

EEGScope es una aplicación web desarrollada como Trabajo de Fin de Grado en Ingeniería Informática en la Universidad de Burgos. Permite trabajar con señales EEG y comparar modelos de Machine Learning y Deep Learning para clasificarlas como TDAH o Control.

El repositorio incluye la aplicación, los scripts de investigación, los resultados experimentales y enlaces a las fuentes consultadas.

**[Página del proyecto](https://eegscope.dev) · [Aplicación web](https://app.eegscope.dev)**

Es un prototipo académico sin validación clínica. La versión publicada depende de que el equipo que aloja los servicios esté encendido y conectado.

## Funcionalidades

- Registro e inicio de sesión, con acceso a los recursos de cada usuario.
- Carga, análisis y reutilización de datasets en CSV.
- Configuración y entrenamiento de modelos ML y DL.
- Historial de experimentos con parámetros, métricas y resultados.
- Predicción sobre el registro de un paciente con modelos propios o de referencia.
- Análisis y entrenamientos en segundo plano, sin bloquear la navegación.

El flujo habitual es **cargar datos → entrenar → consultar resultados → utilizar el modelo**. También se puede predecir con un modelo de referencia sin entrenar uno propio.

## Instalación local

Necesitas **Git y Docker con Docker Compose**, con soporte para contenedores Linux. No hace falta instalar Python ni Node.js por separado para ejecutar la aplicación con Docker.

### 1. Descargar el proyecto

```bash
git clone https://github.com/ajg1005/eeg-adhd-clasificacion.git
cd eeg-adhd-clasificacion
```

### 2. Configurar el entorno

Copia `.env.example` a `.env`:

```powershell
# PowerShell
Copy-Item .env.example .env
```

```bash
# Linux o macOS
cp .env.example .env
```

Edita `.env` y configura:

- `POSTGRES_PASSWORD`: sustituye `cambiame` por una contraseña propia.
- `JWT_SECRET_KEY`: introduce una clave aleatoria de al menos 32 caracteres.

No publiques `.env` ni sus claves. `CLOUDFLARE_TUNNEL_TOKEN` puede quedar vacío para la ejecución local.

### 3. Iniciar la aplicación

Con Docker en ejecución:

```bash
docker compose up --build -d
```

La primera construcción necesita conexión a Internet y puede tardar varios minutos. Docker inicia los servicios y aplica las migraciones de la base de datos.

- **Aplicación:** http://localhost:5173
- **Documentación de la API:** http://localhost:8000/docs

Abre la aplicación y crea una cuenta para comenzar.

Para consultar el estado y los mensajes de los servicios:

```bash
docker compose ps
docker compose logs --tail=100 backend worker
```

Para detenerlos conservando los volúmenes de datos:

```bash
docker compose down
```

## Datos y modelos

### Dataset

Los experimentos utilizan la copia del [EEG Dataset for ADHD disponible en Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd), cuyo conjunto original procede de [IEEE DataPort](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children).

El CSV preparado para los experimentos **no está incluido en el repositorio**. Los scripts esperan encontrarlo en `data/adhdata.csv`; desde la aplicación se carga mediante la interfaz.

### Formato de entrada

El CSV de entrenamiento debe estar separado por comas. Cada fila representa una muestra temporal y contiene valores numéricos para estos 19 canales:

```text
Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz
```

También debe incluir:

| Columna | Contenido |
|---|---|
| `ID` | Identificador del sujeto. |
| `Class` | `ADHD` o `Control`; también se aceptan `1` y `0`, respectivamente. |

Las muestras deben conservar su orden temporal y cada sujeto debe tener una única clase. Se necesitan sujetos de ambas clases suficientes para las particiones configuradas.

Para **predicción**, se carga el registro de un único paciente con los canales requeridos por el modelo. Los datos deben ser compatibles con su configuración; los modelos de referencia utilizan **128 Hz**.

### Modelos

El repositorio incluye modelos de referencia y sus metadatos en `models/ml/` y `models/dl/`.

Los modelos entrenados desde la aplicación se guardan por separado en el volumen Docker `trained_models`. PostgreSQL conserva sus metadatos y su relación con el experimento.

## Evaluación experimental

La validación mantiene separados los sujetos de entrenamiento y prueba: las ventanas de una misma persona no aparecen en ambos grupos.

- **ML:** validación cruzada mediante `StratifiedGroupKFold`.
- **DL:** particiones externas por sujeto y validación interna por sujeto para la parada temprana y la elección del umbral.

Las métricas experimentales se calculan **por ventana** y se resumen mediante la media y la desviación típica entre particiones. En inferencia, la aplicación combina las predicciones de las ventanas para clasificar el registro del paciente.

ML y DL utilizan distintas configuraciones de ventanas y preprocesamiento. Las diferencias de rendimiento no pueden atribuirse únicamente al modelo, ni generalizarse sin una evaluación con datos externos.

## Scripts de investigación

Para ejecutarlos fuera de Docker necesitas **Python 3.13, uv y el CSV preparado**. Ejecuta los comandos desde la raíz del repositorio.

Instalar dependencias:

```bash
uv sync --locked --no-build
```

Entrenar y evaluar:

```bash
uv run --locked --no-build python -m scripts.train_ml
uv run --locked --no-build python -m scripts.train_dl
```

Exportar los modelos finales a partir de las configuraciones seleccionadas:

```bash
uv run --locked --no-build python -m scripts.export_model
uv run --locked --no-build python -m scripts.export_model_dl
```

La exportación puede sustituir los modelos de referencia existentes.

Con el modelo ML y sus metadatos disponibles, calcular la importancia por permutación:

```bash
uv run --locked --no-build python -m scripts.feature_importance
uv run --locked --no-build python -m scripts.feature_importance_xgboost
```

Los análisis utilizan datos de prueba separados por sujeto. Los resultados se guardan en `results/` y las gráficas en `Figuras/`.

## Pruebas y calidad

Las pruebas cubren procesamiento EEG, particiones por sujeto, autenticación, permisos, servicios y endpoints de la API.

Comprobaciones de Python:

```bash
uv sync --locked --no-build
uv run --locked --no-build ruff check backend scripts tests
uv run --locked --no-build pytest -m "not slow" --cov --cov-report=term --cov-report=xml
```

Comprobaciones del frontend, con Node.js 22 y npm:

```bash
cd frontend
npm ci --ignore-scripts
npm run typecheck
npm run lint
npm run build
```

GitHub Actions ejecuta estas comprobaciones y SonarQube Cloud analiza el código. La cobertura publicada corresponde al código Python incluido en la medición, no al frontend.

## Tecnologías y estructura

**Backend:** Python, FastAPI, SQLAlchemy, Alembic y PostgreSQL.  
**Frontend:** React, TypeScript y Vite.  
**Procesamiento y modelos:** NumPy, pandas, SciPy, scikit-learn, XGBoost y TensorFlow/Keras.  
**Ejecución:** Docker Compose, Celery, Redis, Caddy y Cloudflare.

```text
backend/      API, autenticación, recursos y tareas en segundo plano.
frontend/     Interfaz web.
landing/      Página de presentación.
alembic/      Migraciones de la base de datos.
scripts/      Procesamiento, entrenamiento y análisis experimental.
models/       Modelos de referencia y metadatos.
results/      Configuraciones y resultados experimentales.
Figuras/      Gráficas de los experimentos.
notebooks/    Pruebas exploratorias.
tests/        Pruebas unitarias y de integración.
docs/         Fuentes consultadas y documentación complementaria.
```

## Autor y licencia

Desarrollado por **Adrián Jiménez García** como Trabajo de Fin de Grado en la Universidad de Burgos.
El código propio se distribuye bajo la [licencia MIT](LICENSE). Los datasets y las dependencias mantienen sus respectivas licencias y condiciones de uso.
