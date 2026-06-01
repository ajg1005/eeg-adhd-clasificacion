# Análisis del código del TFG

Documento de referencia para memoria y defensa. Recorre **qué hace cada parte, dónde vive y por qué está así**, e identifica **cosas a cambiar/humanizar antes de entrega**. No es exhaustivo línea a línea; busca dar al autor el mapa mental para responder cualquier pregunta del tribunal sin titubear.

> Generado tras revisión del repo con la rama actual (~2026-05-30, ~5 semanas para entrega 6 julio).

---

## 1. Resumen ejecutivo

Aplicación web para clasificación binaria **ADHD vs Control** a partir de señales EEG, con dos pipelines paralelos:

- **Machine Learning clásico** sobre features temporales y espectrales por ventana.
- **Deep Learning** (CNN-1D, CNN-LSTM, EEGNet) sobre ventanas crudas filtradas y normalizadas.

Toda evaluación se hace con **`StratifiedGroupKFold` cross-subject** para evitar leakage de paciente. Existen dos capas claramente separadas:

- `scripts/` — pipeline de investigación reproducible (CLI). Genera los modelos finales que la app sirve.
- `backend/` + `frontend/` — aplicación FastAPI + React que permite explorar dataset, lanzar entrenamientos interactivos, predecir nuevos pacientes y revisar histórico de experimentos (DB Postgres).

Stack: **Python 3.12, FastAPI, SQLAlchemy 2.0 + Alembic, Postgres 16, scikit-learn, XGBoost, TensorFlow/Keras, React 19, Vite, Docker Compose**. CI con GitHub Actions + SonarCloud (badges en README).

---

## 2. Mapa del repo

```
backend/      API FastAPI, servicios, capa de modelado y DB nueva
alembic/      Migraciones de la BD de experimentos (initial schema)
frontend/     SPA React + Vite (4 pestañas: Modelo, Dataset, Entrenamiento, Experimentos, Predicción)
scripts/      Pipeline de research: train ML/DL, export modelo, feature importance
models/       Modelos exportados (final_model.joblib, final_model.keras) + metadatos
results/      Resultados de CV (CSVs y JSON best_model_config)
Figuras/      Matrices de confusión, ROC, curvas de entrenamiento DL
notebooks/    Experimentación preliminar
tests/        Tests pytest (unit + integration)
data/         CSV de dataset (adhdata.csv) — no versionado
docs/         Esta carpeta (licencias.md, enlaces.md, este análisis)
docker-compose.yml, Dockerfile, requirements.txt, alembic.ini, .env.example
```

Filosofía de organización: **research separado de app**. Si mañana se borran `backend/` y `frontend/`, los scripts siguen reproduciendo los resultados desde cero. Si mañana se cambia el modelo exportado, la app lo recoge sin tocar código (solo `models/`).

---

## 3. Pipeline de research (`scripts/`)

### 3.1. Carga y preprocesado (`data_load.py`, `preprocessing.py`)

- `load_dataset(csv_path)`: lee CSV bruto con columnas `ID`, `Class` y 19 canales EEG del sistema 10-20.
- `preprocess_dataset(df)`:
  - Valida columnas obligatorias.
  - Elimina filas con `ID` o `Class` vacíos.
  - **Normaliza `Class` a 0/1** (acepta aliases: "adhd", "tdah", "control", "healthy", "sano", 0, 1). Si encuentra valores desconocidos, raise.
  - Devuelve dataframe limpio + lista de columnas EEG.

**Por qué**: el dataset original puede venir con clases textuales mezcladas y filas corruptas. Esta normalización es **prerequisito** para todo el resto.

### 3.2. Filtrado y normalización por sujeto (`signal_preprocessing.py`)

- `apply_basic_filtering(df, eeg_cols, sfreq=128, lowcut=0.5, highcut=50.0)`:
  - Filtro Butterworth pasabanda orden 4, **por sujeto** (no por bloque) con `filtfilt` (fase cero).
  - Elimina deriva lenta (<0.5 Hz) y ruido por encima de 50 Hz (línea eléctrica).
- `zscore_per_subject(df, eeg_cols)`:
  - Z-score por canal y sujeto (usa media/std del sujeto, no globales).
  - **Importante**: normalizar por sujeto evita que el modelo memorice amplitudes individuales y le obliga a aprender patrones de forma. Solo se usa en el pipeline DL (en ML se confía en `StandardScaler` por fold).

### 3.3. Segmentación en epochs (`epochs.py`)

`create_epochs(df, eeg_columns, epoch_size, step_size)`:
- Agrupa por `ID` (sujeto). Para cada uno, recorre la señal con ventana deslizante (`epoch_size`, `step_size`).
- Devuelve `(x_epochs, y_epochs, groups_epochs)` con shape `(N, epoch_size, n_channels)`, etiqueta heredada del sujeto y vector de grupos para el CV.

**Defaults**:
- ML: `epoch_size=1920` (15 s @128Hz), `step_size=960` (50% overlap).
- DL: `epoch_size=512` (4 s @128Hz), `step_size=256` (50% overlap).

**Por qué dos tamaños distintos**: las features espectrales necesitan ventanas largas para resolución frecuencial decente (∆f = sfreq/nperseg). Las redes 1D aprenden mejor con secuencias más cortas (menos parámetros, más muestras).

### 3.4. Features (`features.py`, `spectral_features.py`, `feature_pipeline.py`)

- **Temporales** (`features.py`): por canal y epoch → mean, std, median, var, min, max, range, q25, q75, iqr, energy, rms. 12 features × 19 canales = **228 features**.
- **Espectrales** (`spectral_features.py`):
  - `welch(signal, fs=sfreq, nperseg)` → PSD.
  - Por banda (delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45): potencia absoluta y relativa.
  - Entropía espectral global.
  - Ratio theta/beta (clínicamente relevante en TDAH — biomarcador candidato discutido en literatura).
  - Frecuencia beta media solo en O1, O2 (occipitales — específico de literatura).
- **`feature_pipeline.build_features_from_epochs`** combina ambas según `feature_mode` ("temporal" / "spectral" / "combined"). El **default es "combined"**.

### 3.5. Splits (`split.py`)

- `make_group_kfold_splits`: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. **Punto clave del TFG**: estratifica por clase a nivel epoch pero **garantiza que un sujeto no aparece en train y test del mismo fold**. Sin esto, el modelo memoriza la señal del paciente y rinde irrealmente bien (data leakage por sujeto, error típico en literatura EEG).
- `make_group_shuffle_split`: split sujeto-a-sujeto estratificado por clase. Usado en DL para sacar validación interna desde el train de cada fold outer.

### 3.6. Modelos

#### ML (`pipeline.py`, `ml_model_registry.py`)

5 modelos disponibles: `logistic_regression`, `rbf_svc`, `knn`, `random_forest`, `xgboost`. Todos envueltos en `sklearn.Pipeline` (con `StandardScaler` cuando aplica). Defaults vienen del registro central `ml_model_registry.MODEL_SPECS`.

`UI_MODEL_NAMES = ("rbf_svc", "random_forest", "xgboost")` — solo se exponen 3 en la UI; los otros 2 existen para baseline comparativo en research.

#### DL (`tf_models.py`)

- **EEGNet**: variante ligera con `Conv1D` + `SeparableConv1D` + `BatchNorm` + `AvgPool` + `SpatialDropout`.
- **CNN 1D**: 3 bloques Conv1D progresivos (16→32→64 filtros) + dense.
- **CNN-LSTM**: 2 bloques Conv1D + LSTM bidireccional + dense. Captura temporalidad explícitamente.

Todos terminan en `Dense(1, sigmoid)` para binario, con regularización L2 ligera.

### 3.7. Entrenamiento ML (`train_ml.py`)

Flujo:
1. Carga → preprocess → epochs (default 15 s).
2. Features combined (228 cols).
3. 5-fold CV cross-subject sobre todos los modelos.
4. Recopila métricas por fold + predicciones OOF (out-of-fold).
5. Resume por modelo (media ± std).
6. **Elige mejor por `F1_mean_cv`** → guarda `results/best_model_config.json`.
7. Genera figuras: comparación entre modelos, confusion matrix y ROC del mejor modelo OOF.

### 3.8. Entrenamiento DL (`train_dl.py`)

Más complejo porque cada red necesita más infra:
1. Carga → preprocess → **filtrado + zscore por sujeto** (NO se hace en ML).
2. Epochs cortas (4 s).
3. 5-fold CV outer cross-subject. **Dentro de cada fold**, otro split sujeto-a-sujeto (80/20) para validación → early stopping y selección de threshold.
4. Para cada modelo:
   - `set_seed` por fold (reproducibilidad).
   - `Adam(lr=3e-4, clipnorm=1.0)`, loss `BinaryCrossentropy(label_smoothing=0.05)`.
   - Callbacks: `EarlyStopping(monitor=val_loss, patience=4, restore_best_weights=True)` + `ReduceLROnPlateau`.
   - `find_best_threshold(y_val, y_val_score)` busca threshold óptimo en validación → aplicado en test.
5. Guarda curvas de entrenamiento por fold (PNG en `Figuras/training_curves_tf/`).
6. Mismo proceso de mejor modelo + figuras + JSON config.

**Por qué label smoothing 0.05**: reduce overconfidence del modelo (etiqueta 1 → distribución [0.05, 0.95] en loss). Útil cuando hay ruido en etiquetas o batches pequeños.

### 3.9. Export de modelos (`export_model.py`, `export_model_dl.py`)

Toman el `best_model_config.json` correspondiente, **reentrenan el modelo seleccionado sobre TODO el dataset** (no CV) y lo serializan a `models/{ml|dl}/final_model.{joblib|keras}` junto a `model_metadata.json` y `model_metrics.json`. Es lo que carga el backend para predicción.

**Por qué reentrenar sobre todo**: el CV sirve para estimar generalización; el modelo desplegado debe aprovechar el 100% de los datos.

### 3.10. Feature importance (`feature_importance.py`)

`sklearn.inspection.permutation_importance` sobre test held-out cross-subject. Identifica qué features (canal + banda) más afectan al rendimiento si se aleatorizan. Resultado se agrega por canal para la UI.

### 3.11. Constantes compartidas (`constants.py`)

**Fuente única de verdad** que `backend/constants.py` re-exporta:
- `RANDOM_STATE = 42`.
- `REQUIRED_EEG_COLUMNS`: lista de los 19 canales 10-20.
- `REQUIRED_COLUMNS`: anterior + `["Class", "ID"]`.
- `normalize_class_to_int` / `normalize_class_to_label`: aliases ADHD/Control.

---

## 4. Backend (FastAPI)

### 4.1. Estructura

```
backend/
  main.py              # FastAPI app + CORS + routers + mount /figures
  config.py            # paths + construcción de DATABASE_URL
  constants.py         # re-exporta scripts.constants
  routers/             # endpoints HTTP (uno por dominio)
  services/            # lógica de negocio (sin conocer HTTP)
  schemas/             # modelos Pydantic (validación + docs OpenAPI)
  modeling/            # factories de modelos ML/DL para la app
  db/                  # NUEVO: SQLAlchemy engine + models + repository
```

Separación clásica **routers → services → modeling/db**. Routers son finos (parsing/HTTP/errores), services hacen el trabajo, modeling/db son helpers de bajo nivel.

### 4.2. Routers

- **`health.py`** → `GET /health` (liveness simple).
- **`models.py`** →
  - `GET /models` lista modelos disponibles para selector.
  - `GET /model/catalog` lista modelos candidatos + parámetros típicos para UI de entrenamiento.
  - `GET /model/info?model_id=...` info + métricas del modelo cargado.
  - `GET /model/figures?model_id=...` lista de PNGs servidos en `/figures/...`.
- **`prediction.py`** →
  - `POST /validate` valida CSV contra el modelo seleccionado.
  - `POST /predict` lanza predicción.
- **`training_router.py`** (prefix `/training`) →
  - `GET /training/options` valores por defecto + opciones de hiperparámetros.
  - `POST /training/dataset/stats` analiza un CSV (filas, columnas, distribución de clases, lista de pacientes).
  - `POST /training/run` entrena cross-subject CV con el modelo elegido y devuelve métricas + folds + feature importance.
- **`experiments.py`** (NUEVO, sin prefix) →
  - `GET /experiments` lista paginada con filtros `model_type`, `model_name`.
  - `GET /experiments/{id}` detalle con folds embebidos.

### 4.3. Servicios

- **`csv_service.py`** — leer `UploadFile` y convertir a DataFrame.
- **`model_service.py`** — fachada sobre los predictores (`get_predictor`, `predict_dataframe`).
- **`training_data.py`** — orquesta preprocess + epochs + features. Reutiliza scripts.
- **`training_runners.py`** — corazón del entrenamiento desde la app:
  - `run_ml_cross_subject_cv` y `run_dl_cross_subject_cv`. Mismo flujo que en `scripts/train_*.py` pero parametrizado y devolviendo métricas listas para JSON.
  - `_safe_feature_importance_for_fold` corre permutation_importance solo en el primer fold (caro) con submuestreo estratificado.
  - `_release_keras_model` libera memoria entre folds DL (`clear_session`, `gc.collect`, limpia cache LRU de predictores).
- **`training_service.py`** — punto de entrada del endpoint `/training/run`:
  1. Valida tipos/params.
  2. Lee CSV, prepara epochs.
  3. Llama al runner correspondiente.
  4. Calcula métricas globales, classification_report, confusion_matrix, patient_results.
  5. **Guarda el experimento en BD** (`save_experiment`) y devuelve `experiment_id` en la respuesta.

### 4.4. Modeling

- **`model_factory.py` / `dl_factory.py`** — wrappers que combinan defaults del registro con params del usuario y delegan en `scripts.pipeline.create_ml_model` / `scripts.tf_models.build_model`.
- **`model_catalog.py`** — devuelve el catálogo de modelos UI con sus defaults para `GET /model/catalog`.
- **`predictors.py`** — `MLPredictor` y `DLPredictor` con `@cached_property` para artifacts y `@lru_cache` sobre `get_predictor(model_id)`. **Importante**: solo se carga el `.joblib`/`.keras` la primera vez; las siguientes peticiones reutilizan el modelo en memoria.
- **`common.py`** — utilidades compartidas (`validate_eeg_dataframe`, `prepare_features_from_dataframe`, etc).

### 4.5. Persistencia (capa nueva)

Documentación detallada en sección 5.

### 4.6. Config

`config.py` construye `DATABASE_URL` así:
1. Si `DATABASE_URL` está en env → la usa (override explícito).
2. Si hay `POSTGRES_PASSWORD` → construye `postgresql+psycopg://user:pass@host:port/db` con las demás `POSTGRES_*`.
3. Si nada → fallback a SQLite local (`./experiments.db`) para que scripts/tests funcionen sin Docker.

**Por qué este patrón**: aplica el principio de "12-factor app" (config por env vars), permite el mismo código en Docker y en local, y deja Postgres opcional.

---

## 5. Persistencia de experimentos (DB)

### 5.1. Modelo de datos

Tres tablas:

#### `datasets`
- `id`, `dataset_hash` (sha256 del CSV, único), `filename`, `rows`, `columns`, `n_subjects`, `class_distribution` (JSON), `eeg_columns` (JSON), `created_at`.
- **Dedupe**: el mismo CSV subido N veces genera UN solo dataset row (búsqueda por hash).

#### `experiments`
- FK a `datasets`. Almacena: tipo/nombre de modelo, modo de evaluación, tiempo de entrenamiento, métricas globales (accuracy, balanced_accuracy, precision, recall, f1), tres JSON con params (eeg, model, training), confusion_matrix (JSON), classification_report (JSON).
- Indexes en `dataset_id`, `model_type`, `model_name` para filtros rápidos.

#### `experiment_folds`
- FK a `experiments`. Una fila por fold: métricas + `n_train/val/test_subjects` + `best_threshold` (DL only).
- Index en `experiment_id`.

### 5.2. Flujo de guardado

`backend/services/training_service.py:139` llama a `save_experiment` al final de `run_training`:

1. Calcula sha256 del CSV → busca o crea `Dataset`.
2. Crea `Experiment` y le asigna el `dataset_id`.
3. `session.flush()` para obtener `experiment.id`.
4. Inserta N `ExperimentFold` con ese id.
5. `commit`.
6. Devuelve `experiment_id` que se inyecta en la respuesta API.

### 5.3. Engine + sesión

`db/engine.py`:
- `create_engine(DATABASE_URL, ...)` con `check_same_thread=False` solo en SQLite.
- `SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)`.
- `init_db()` con `Base.metadata.create_all` — **solo usado por tests**. Producción usa Alembic.

### 5.4. Schemas Pydantic (`schemas/experiments.py`)

- `ExperimentDatasetResponse`, `ExperimentSummaryResponse`, `ExperimentFoldResponse`, `ExperimentDetailResponse` (extiende summary con configs y folds), `ExperimentsListResponse`.
- Todos heredan de `OrmSchema` que tiene `model_config = ConfigDict(from_attributes=True)` para mapear ORM → API.

### 5.5. Frontend

`ExperimentsView.jsx`:
- Lista paginada de experimentos con tabla (ID, fecha, modelo, dataset, F1, Balanced).
- Click en fila → carga detalle con métricas globales + dataset + configuración JSON.
- `useEffect` separados para lista y detalle.

### 5.6. Alembic

`alembic/env.py` importa `DATABASE_URL` y `Base.metadata` del propio backend. La primera migración (`20260527_0001_initial_schema.py`) crea las tres tablas con índices y FKs.

En Docker el backend ejecuta `alembic upgrade head && uvicorn ...` como comando. `db` service tiene healthcheck con `pg_isready` y backend espera con `depends_on: condition: service_healthy`.

---

## 6. Frontend (React + Vite)

### 6.1. Arquitectura

```
src/
  main.jsx                # entrypoint
  App.jsx                 # shell, tabs, routing por estado
  api.js                  # cliente HTTP del backend
  config/tabs.js          # configuración de pestañas
  propTypes.js            # PropTypes compartidos
  utils/formatters.js     # helpers de formato
  hooks/
    useInferenceController.js  # estado de Predicción + Modelo
    useTrainingDataset.js      # estado del CSV de entrenamiento
  components/
    AppHeader, Tabs, ModelSelector, ModelView, DatasetView,
    TrainingView, PredictionView, ExperimentsView
    training/  (TrainingEegParamsPanel, TrainingModelPanel, TrainingResultsPanel)
```

### 6.2. Estado global por hooks

Sin Redux/Zustand. Dos custom hooks en `App.jsx`:
- `useInferenceController` — estado de tabs, modelo seleccionado, archivo de predicción, resultados de predicción, validación, métricas.
- `useTrainingDataset` — CSV de entrenamiento + stats analizadas.

`App.jsx` renderiza la vista correspondiente según `controller.activeTab`. Hay una guardia contra cambio de tab si hay un entrenamiento en curso (`trainingInProgress` + `beforeunload` para avisar al cerrar la ventana).

### 6.3. Pestañas

1. **Modelo** — `ModelSelector` + `ModelView` con métricas, gráficos (Recharts), figuras del CV, catálogo de modelos.
2. **Dataset entrenamiento** — sube CSV, analiza, muestra stats + tabla de pacientes filtrable. Alerta si faltan columnas.
3. **Entrenamiento** — panel de hiperparámetros EEG + modelo + entrenamiento, botón "Entrenar", `TrainingResultsPanel` con métricas, fold-by-fold, patient_results y feature importance.
4. **Experimentos** (NUEVA) — histórico de entrenamientos guardados en DB.
5. **Predicción** — sube CSV de un paciente, valida, predice, muestra clase final + reparto de epochs (con gráfica).

### 6.4. Cliente HTTP

`api.js`:
- `API_BASE_URL` desde `import.meta.env.VITE_API_BASE_URL` (Vite inyecta build-time).
- Cada endpoint es una función `async/await` con `fetch`. Helper `readError` extrae el `detail` del JSON de error de FastAPI.
- Llamadas a `/training/run` usan `FormData` con archivo + JSON-stringified params.

---

## 7. Infra y herramientas

### 7.1. Docker

- **`backend/Dockerfile`**: `python:3.12-slim`, usuario no-root `app` (uid 1001), `WORKDIR /app`, instala con `--no-cache-dir --only-binary=:all:` (forzar wheels, evitar compilación en build). Copia `backend/`, `scripts/`, `models/`, `Figuras/`, `alembic/`, `alembic.ini`. `EXPOSE 8000`.
- **`docker-compose.yml`**:
  - `db` (Postgres 16-alpine) con `env_file: .env`, healthcheck `pg_isready`.
  - `backend` ejecuta `alembic upgrade head && uvicorn` antes de servir. Espera al healthcheck de `db`. Monta `./results:/app/results`.
  - `frontend` (Vite dev server) con `VITE_API_BASE_URL` configurable.

### 7.2. Seguridad de secretos

- `.env` con `POSTGRES_*` está en `.gitignore`.
- `.env.example` versionado con placeholder `PASSWORD` (para documentar formato).
- Docker compose NO referencia `POSTGRES_PASSWORD` con default → Sonar no detecta secretos hardcoded.

### 7.3. Tests

`pytest` con dos niveles:
- `tests/unit/` — funciones puras (constants, features, preprocessing, spectral, split, modelado, services).
- `tests/integration/` — endpoints (`TestClient` de FastAPI). Cubren health, models, prediction, training. Hay un test de end-to-end de entrenamiento que verifica que se guarda en DB.
- `conftest.py` — setea `DATABASE_URL` a SQLite temporal con UUID único por sesión, llama `init_db()` para crear tablas. Fixtures para CSV de muestra y factory de DataFrame EEG sintético.

### 7.4. CI + calidad

- GitHub Actions ejecuta tests y reporta a SonarCloud (badges en README).
- `pytest.ini`, `sonar-project.properties` configurados.
- ESLint/PropTypes en frontend.

---

## 8. Puntos fuertes a defender (qué te separa del TFG medio)

| Punto | Por qué importa |
|---|---|
| **Cross-subject CV con `StratifiedGroupKFold`** | Evita el error #1 en clasificación EEG: data leakage por paciente. Sin esto, las accuracies parecen 95%+ y son irreales. |
| **Comparativa ML vs DL con misma metodología** | No es "entreno una CNN y ya". Hay 5 modelos ML + 3 DL evaluados igual. |
| **Threshold óptimo en validación, aplicado en test (DL)** | `find_best_threshold` evita el sesgo de usar 0.5 ciego. |
| **Permutation importance sobre test held-out** | Metodológicamente correcto (no sobre train). Agregado por canal aporta interpretabilidad clínica. |
| **`features.combined` (temporal + espectral)** | Solo espectral o solo temporal pierde información. Combined es lo que usa la literatura actual. |
| **Dedupe de datasets por sha256** | Trazabilidad real: sabes qué CSV produjo qué experimento. |
| **CI + SonarCloud + tests integration + unit** | Madurez de proceso, no solo de código. Muy raro en TFGs. |
| **Postgres + Alembic + Docker Compose** | Stack "production grade" sin caer en sobreingeniería. |
| **Capa `scripts/` reproducible separada de la app** | Si la app cae, los modelos siguen siendo reproducibles desde cero. |
| **`@cached_property` + `@lru_cache` en predictores** | Optimización real: no recargar `.keras` en cada request. |

---

## 9. Riesgos para defensa (lo que el tribunal puede atacar)

### 9.1. Código que "huele a IA"

| Lugar | Problema | Cómo defenderlo / arreglarlo |
|---|---|---|
| `backend/config.py:_database_url` (versión previa) | Walrus operator, `urllib.parse.quote`, fail-fast. | Ya simplificado. |
| `db/repository.py` | `selectinload` con eager loading. | Defender: "evitar N+1 queries al servir lista de experimentos". |
| `db/models.py` | Patrón `Mapped[...]` + `mapped_column`. | Es la sintaxis **oficial recomendada** de SQLAlchemy 2.0 desde 2023. |
| `schemas/experiments.py` | `OrmSchema` con `from_attributes=True`. | Equivalente moderno a `Config: orm_mode = True` de Pydantic v1. |
| `ExperimentsView.jsx:23, 38, 49` | `await Promise.resolve()` y `setTimeout(loadExperiments, 0)` no hacen nada útil. | **BORRAR** antes de defensa. Es ruido sin propósito. |
| `docker-compose.yml` | Healthcheck `pg_isready` y `condition: service_healthy`. | Es estándar. Defender: "Postgres tarda en aceptar conexiones tras arrancar el proceso; sin healthcheck, alembic upgrade falla con connection refused". |

### 9.2. Limitaciones conocidas (mencionar como honestidad, no esconder)

- **Solo soporta el montaje 10-20 de 19 canales** (`REQUIRED_EEG_COLUMNS` hardcoded). Datasets con otros montajes (más/menos canales o nombres distintos tipo `EEG-Fp1`) no funcionan sin refactor.
- **Solo binario ADHD vs Control**. Multi-clase (ej. subtipos de TDAH) requeriría refactor en métricas, modelos y UI.
- **Frecuencia de muestreo asumida 128 Hz** en defaults. Otros sfreq requieren ajustar `epoch_size`, `step_size`, `nperseg` desde la UI.
- **No hay autenticación**. App pensada para uso local académico.
- **`save_experiment` síncrono dentro del endpoint**: si la DB cae después de 10 min de CV, se pierden los resultados. Aceptable para TFG, mejorable con cola asíncrona en producción.

### 9.3. Preguntas probables del tribunal y respuestas

> "¿Por qué `StratifiedGroupKFold` y no `KFold`?"
>
> Porque las epochs del mismo paciente están correlacionadas (provienen de la misma señal continua). Si un paciente aparece en train y test, el modelo aprende características individuales y rinde irrealmente bien. `Group` garantiza separación por paciente; `Stratified` mantiene proporción de clases en cada fold.

> "¿Por qué normalizas por sujeto en DL y no en ML?"
>
> En ML uso `StandardScaler` dentro del `Pipeline`, ajustado por fold, sobre las features ya extraídas. En DL la red entrena sobre la señal cruda; normalizar por sujeto (z-score) elimina diferencias de amplitud individuales y obliga a la red a aprender patrones de forma, no de escala. Sin esto, batches con sujetos de amplitud muy distinta confunden el aprendizaje.

> "¿Por qué Postgres y no SQLite siendo un TFG?"
>
> Quería que el sistema fuese realista para un uso real, no solo de demo. SQLite tiene limitaciones de concurrencia y serialización JSON nativa. Postgres ofrece tipos nativos, concurrencia y se puede desplegar igual en producción. Para tests uso SQLite por velocidad e aislamiento — patrón estándar.

> "¿Por qué Alembic si solo tienes una migración?"
>
> Para versionar cambios futuros del esquema sin perder datos. La alternativa (`create_all`) recrea desde cero y no soporta evolución. En tests sigo usando `create_all` por velocidad porque el esquema se monta vacío en cada sesión.

> "¿Cómo eliges el threshold del modelo DL?"
>
> Para cada fold, dentro del train hago un sub-split sujeto-a-sujeto 80/20 para validación interna. Sobre esa validación busco el threshold que maximiza balanced_accuracy (desempate por F1) en una rejilla 0.2–0.8 con 61 puntos. Ese threshold se aplica al test del fold. Evita sesgar el threshold usando el test.

> "¿Cómo evitas overfitting en DL?"
>
> `EarlyStopping` con `restore_best_weights` (patience 4 sobre val_loss), `ReduceLROnPlateau`, dropout 0.4, label smoothing 0.05, regularización L2 ligera, gradient clipping (`clipnorm=1.0`).

> "¿Por qué `feature_importance` solo en un fold y no en todos?"
>
> `permutation_importance` con scoring `f1_weighted` es caro: por feature, recalcula la métrica permutando esa columna. Multiplicado por 5 folds × N repeats × N features se hace inviable en tiempo de respuesta de una API. Hago el cálculo solo en el primer fold con submuestreo estratificado a 80 epochs y 1 repeat. Asumo que el ranking de importancia es relativamente estable entre folds. Para análisis riguroso, `scripts/feature_importance.py` lo calcula completo offline.

---

## 10. Cosas concretas a cambiar / humanizar antes de entrega

Lista priorizada (de más impacto a menos):

### Prioridad alta (1-2 horas total)

1. **Borrar el ruido AI-generated en `ExperimentsView.jsx`**:
   - Líneas 23, 49: `await Promise.resolve()` — no hace nada.
   - Líneas 38-40: `setTimeout(loadExperiments, 0)` — equivalente a llamar directamente.
   - Sustituir por llamadas directas. No tienes cómo defender estas líneas.

2. **`datetime.utcnow` deprecado** en `backend/db/models.py:24,36`:
   ```python
   # Antes
   default=datetime.utcnow
   # Después
   from datetime import datetime, timezone
   default=lambda: datetime.now(timezone.utc)
   ```
   En Python 3.12 emite warning, en 3.13 desaparece.

3. **URL-encode de la password en `config.py`**:
   Si tu `POSTGRES_PASSWORD` contiene `@`, `:`, `/`, `#`, `%`, `?`, la URL se rompe. Fix:
   ```python
   from urllib.parse import quote
   ...
   DATABASE_URL = f"postgresql+psycopg://{pg_user}:{quote(pg_pass, safe='')}@{pg_host}:{pg_port}/{pg_db}"
   ```
   O simplemente documenta en `.env.example` que la password debe ser alfanumérica.

4. **`getExperiments` en `api.js:106` usa error genérico**, mientras `getExperimentDetail` usa `readError`. Inconsistencia visible:
   ```javascript
   if (!response.ok) {
     throw new Error(await readError(response, "No se pudo cargar el historial de experimentos"));
   }
   ```

### Prioridad media (cosmético / defensable)

5. **`_class_distribution` duplicado** en `backend/db/repository.py:141` y `backend/services/training_data.py:106`. Extraer a un helper compartido (ej. `backend/services/dataset_utils.py`).

6. **`_fold_from_result` defaults `fold=0`** en `repository.py:120`. Si la key cambia, todos los folds quedan numerados 0. Mejor:
   ```python
   fold=int(fold["fold"]),  # KeyError si falta, mejor que silenciarlo
   ```

7. **`init_db()` es dead code en producción**, solo usado por tests. Añadir docstring explícito:
   ```python
   def init_db() -> None:
       """Crea las tablas usando metadata. Solo para tests; producción usa Alembic."""
       Base.metadata.create_all(bind=engine)
   ```

8. **`alembic.ini:4`** tiene `sqlalchemy.url = postgresql+psycopg://` (placeholder vacío). Mejor dejar comentario:
   ```ini
   # URL la setea env.py desde backend.config.DATABASE_URL
   sqlalchemy.url = driver://user:pass@localhost/dbname
   ```

### Prioridad baja (solo si sobra tiempo)

9. **`ExperimentsView` no limpia `selectedId` si el experimento desaparece** del listado tras refresh. Cuando se añada DELETE, esto se notará. Para TFG: no urgente.

10. **`save_experiment` síncrono**: si falla, se pierde el entrenamiento. Wrap en try/except y devolver el `result` aunque save haya fallado, con flag `saved: false`. Más robusto, pero cambia el contrato de respuesta.

11. **`Dockerfile` `chmod -R 755 /app`** deja los archivos de migración Alembic ejecutables. No causa bug, pero rompe principio de mínimo privilegio. Defensa: "los archivos Python se importan, no se ejecutan directamente".

12. **`backend/routers/training_router.py:48`** cambió el orden de parámetros (`file` después de `model_type`/`model_name`). No rompe nada porque son `Form()`, pero si OpenAPI clientes externos usaban orden posicional, regenerar.

---

## 11. Glosario rápido para defensa

| Término | Significado |
|---|---|
| **Cross-subject CV** | Validación cruzada donde un mismo paciente no aparece en train y test del mismo fold. |
| **StratifiedGroupKFold** | Como GroupKFold pero manteniendo proporción de clases en cada fold. |
| **OOF (out-of-fold)** | Predicciones de un sample obtenidas cuando ese sample estaba en test. Permite calcular métricas globales sobre todo el dataset usando solo modelos que no lo vieron en train. |
| **Welch** | Método para estimar PSD (potencia espectral) promediando periodogramas de segmentos solapados. |
| **theta/beta ratio** | Cociente de potencia banda theta (4-8 Hz) entre beta (13-30 Hz). Biomarcador candidato de TDAH discutido en literatura. |
| **Permutation importance** | Mide importancia de una feature aleatorizando su columna y viendo cuánto cae la métrica. |
| **Label smoothing** | Reemplaza etiquetas binarias [0,1] por [α, 1-α] (ej. 0.05/0.95) en el loss para reducir overconfidence. |
| **Early stopping** | Parar entrenamiento cuando una métrica de validación no mejora durante N epochs. |
| **Z-score per subject** | Normalizar señal de cada sujeto restando su media y dividiendo por su std (no globales). |
| **Filtfilt** | Filtro digital aplicado dos veces (forward + backward) para eliminar el desfase introducido por filtros causales. |
| **`@lru_cache`** | Decorator de Python que cachea resultados de una función por argumentos. Aquí evita recargar modelo de disco. |
| **`@cached_property`** | Como `@property` pero el resultado se computa una vez por instancia y se cachea. |
| **`selectinload`** | Estrategia de SQLAlchemy para cargar relaciones con una SELECT IN extra (no N+1). |
| **`from_attributes=True`** | Pydantic v2: permite construir el modelo desde objetos con atributos (ORM), no solo dicts. |

---

## 12. Conclusión

El proyecto es **claramente sobresaliente en producto técnico**. Lo que decide la nota desde aquí es la **memoria** y la **defensa**: 5-6 semanas son suficientes para llegar bien si te centras en escribir y preparar respuestas, no en seguir añadiendo features.

Lo que SÍ vale la pena tocar antes de entrega:
- Los puntos de prioridad alta de la sección 10 (1-2 h).
- Estudiarte cada archivo de la capa nueva (DB + experiments) para defenderlos línea por línea.

Lo que NO:
- Más features. Cada hora extra en código resta tiempo a la memoria.
- Refactors arquitectónicos. Lo que hay es coherente y defendible.
