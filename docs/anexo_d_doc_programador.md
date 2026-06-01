# Apéndice D — Documentación técnica de programación

## D.1. Introducción

Este apéndice está orientado a un desarrollador que quiera **comprender,
modificar, ejecutar o extender** el proyecto **EEG ADHD Classifier**.
Incluye la estructura de directorios del repositorio, el manual del
programador con instrucciones de instalación y desarrollo, y la
referencia a la documentación interna del código.

El proyecto está alojado en el repositorio público
<https://github.com/ajg1005/eeg-adhd-clasificacion> y se distribuye bajo
licencia MIT.

---

## D.2. Estructura de directorios

A continuación se describe el contenido de cada carpeta del repositorio.
Se distinguen tres bloques: **código de la aplicación** (backend, frontend,
db, scripts), **artefactos generados** (models, results, Figuras) y
**documentación y configuración** (docs, ficheros raíz).

```
eeg-adhd-clasificacion/
├── backend/                  # API FastAPI servida en producción
│   ├── main.py               # FastAPI app + CORS + montaje de routers
│   ├── config.py             # construcción de DATABASE_URL y settings
│   ├── constants.py          # re-exporta constantes del pipeline
│   ├── Dockerfile            # imagen de runtime del backend
│   ├── routers/              # endpoints HTTP (un router por dominio)
│   ├── services/             # lógica de negocio sin conocer HTTP
│   ├── schemas/              # esquemas Pydantic (validación + OpenAPI)
│   ├── modeling/             # fachadas y predictores de modelos ML/DL
│   └── db/                   # capa de persistencia (engine, ORM, repo)
│
├── frontend/                 # SPA React 19 + Vite
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx          # entrypoint React
│       ├── App.jsx           # shell, pestañas
│       ├── api.js            # cliente HTTP del backend
│       ├── components/       # componentes de cada pestaña
│       ├── hooks/            # hooks de estado (inference, training)
│       ├── config/tabs.js    # configuración de pestañas
│       ├── utils/formatters.js
│       └── propTypes.js
│
├── scripts/                  # pipeline de research (CLI, no se sirve)
│   ├── data_load.py          # carga del CSV bruto
│   ├── preprocessing.py      # validación y limpieza
│   ├── signal_preprocessing.py # filtrado + z-score por sujeto (DL)
│   ├── epochs.py             # segmentación en epochs cross-subject
│   ├── features.py           # features temporales por canal
│   ├── spectral_features.py  # features espectrales por banda EEG
│   ├── feature_pipeline.py   # combinación de features (temporal/espectral/combined)
│   ├── split.py              # StratifiedGroupKFold y splits cross-subject
│   ├── pipeline.py           # factory de modelos ML (5 modelos)
│   ├── ml_model_registry.py  # registro central de specs ML
│   ├── tf_models.py          # 3 redes DL (CNN-1D, CNN-LSTM, EEGNet)
│   ├── train_ml.py           # entrenamiento ML offline con CV
│   ├── train_dl.py           # entrenamiento DL offline con CV
│   ├── export_model.py       # exporta el mejor modelo ML
│   ├── export_model_dl.py    # exporta el mejor modelo DL
│   ├── feature_importance.py # permutation importance offline
│   └── constants.py          # fuente de verdad (canales, RANDOM_STATE, aliases)
│
├── alembic/                  # migraciones de la BD
│   ├── env.py
│   └── versions/
│       └── 20260527_0001_initial_schema.py
├── alembic.ini
│
├── tests/                    # tests pytest
│   ├── conftest.py
│   ├── fixtures/
│   ├── unit/                 # 12+ tests unitarios
│   └── integration/          # 4+ tests de endpoints con TestClient
│
├── models/                   # modelos exportados servidos por la API
│   ├── ml/                   # final_model.joblib + metadatos
│   └── dl/                   # final_model.keras + metadatos
│
├── results/                  # resultados de CV (CSVs y JSON)
│   └── best_model_config.json
│
├── Figuras/                  # PNGs generados (CM, ROC, curvas DL, etc.)
│   └── training_curves_tf/
│
├── notebooks/                # experimentación preliminar
│   └── tuning_ligero_rf_svc.ipynb
│
├── data/                     # CSV de dataset (NO versionado)
│
├── docs/                     # documentación del proyecto
│   ├── analisis_codigo.md
│   ├── anexo_a_plan_proyecto.md
│   ├── anexo_b_requisitos.md
│   ├── anexo_c_diseno.md
│   ├── anexo_d_doc_programador.md   ← este documento
│   ├── enlaces.md
│   └── licencias.md
│
├── .github/workflows/        # CI con GitHub Actions
├── docker-compose.yml        # despliegue completo (backend + frontend + db)
├── requirements.txt          # dependencias backend producción
├── requirements-dev.txt      # tests, linter
├── pytest.ini                # configuración pytest
├── sonar-project.properties  # SonarCloud
├── .env.example              # plantilla de variables de entorno
├── LICENSE                   # MIT License
└── README.md                 # información general y arranque rápido
```

### D.2.1. Filosofía de la organización

- **Separación research / producto**: la carpeta `scripts/` contiene el
  pipeline de investigación reproducible (CLI), mientras que `backend/`
  contiene la aplicación servida. Los modelos exportados a `models/`
  son el puente entre ambos: los scripts los generan, el backend los
  consume.
- **Capas estrictas en el backend**: los `routers/` no conocen detalles
  de modelo ni de BD; los `services/` no conocen HTTP; los predictores
  (`modeling/`) ni la persistencia (`db/`) conocen los routers.
- **Fuente única de verdad**: `scripts/constants.py` define `RANDOM_STATE`,
  los nombres de los 19 canales 10-20, las columnas obligatorias y los
  aliases ADHD/Control. `backend/constants.py` re-exporta sin duplicar.

---

## D.3. Manual del programador

### D.3.1. Requisitos previos

Para trabajar con el proyecto se necesita uno de estos dos entornos:

#### Opción A — Entorno con Docker (recomendado)

- **Docker Desktop** o equivalente con Docker Engine + Docker Compose.
- **8 GB de RAM libres** mínimo (los modelos DL son intensivos).
- Navegador moderno (Chrome, Firefox, Edge) para acceder a la SPA.

#### Opción B — Entorno local sin Docker

- **Python 3.12+**.
- **Node.js 20+** y npm.
- **PostgreSQL 16** (opcional: si falta, el backend usa SQLite local
  automáticamente — útil para tests y desarrollo).

### D.3.2. Instalación y arranque

#### Arranque completo con Docker Compose

```bash
git clone https://github.com/ajg1005/eeg-adhd-clasificacion.git
cd eeg-adhd-clasificacion
cp .env.example .env             # editar la contraseña de la BD
docker compose up --build
```

Esto levanta los **tres servicios**:

| Servicio  | Imagen base         | Puerto host | Función                                   |
|-----------|---------------------|-------------|-------------------------------------------|
| `db`      | `postgres:16-alpine`| 5432        | Base de datos PostgreSQL con healthcheck. |
| `backend` | construida local    | 8000        | FastAPI + Uvicorn. Ejecuta `alembic upgrade head` antes de aceptar tráfico. |
| `frontend`| construida local    | 5173        | SPA Vite servida en modo dev.             |

Acceso:

- Frontend: <http://localhost:5173>
- API y documentación interactiva: <http://localhost:8000/docs>

#### Arranque local del backend

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn backend.main:app --reload --port 8000
```

Sin variables de entorno, el backend usará SQLite local automáticamente
(`./experiments.db`), suficiente para desarrollo y tests.

#### Arranque local del frontend

```bash
cd frontend
npm install
npm run dev
```

### D.3.3. Variables de entorno

Las variables se cargan desde un fichero `.env` en la raíz del proyecto.
El fichero `.env.example` documenta el formato:

```dotenv
POSTGRES_DB=eeg_adhd
POSTGRES_USER=eeg_app
POSTGRES_PASSWORD=PASSWORD    # cambiar a una contraseña segura
```

Variables adicionales reconocidas por el backend:

| Variable               | Valor por defecto                | Descripción                                                                 |
|------------------------|----------------------------------|-----------------------------------------------------------------------------|
| `DATABASE_URL`         | (autoconstruida)                 | Override completo de la URL de conexión a BD. Si se define, ignora las `POSTGRES_*`. |
| `POSTGRES_HOST`        | `db` (Docker) / `localhost`      | Host de PostgreSQL.                                                          |
| `POSTGRES_PORT`        | `5432`                           | Puerto de PostgreSQL.                                                        |
| `VITE_API_BASE_URL`    | `http://127.0.0.1:8000`          | URL del backend tal y como la verá el navegador (la inyecta Vite en build).  |

> **Importante**: si la contraseña de la BD contiene caracteres reservados
> de URL (`@`, `:`, `/`, `#`, `%`, `?`), se debe usar `DATABASE_URL` con
> *percent-encoding* explícito o evitar dichos caracteres en la
> contraseña.

### D.3.4. Tests y calidad

#### Ejecutar tests

```bash
pytest                            # toda la suite
pytest tests/unit                 # solo unitarios
pytest tests/integration          # solo integración
pytest --cov=backend --cov=scripts # con cobertura
```

Los tests están organizados en dos niveles:

- **`tests/unit/`**: funciones puras (constantes, features temporales y
  espectrales, preprocesado, splits cross-subject, modelado, servicios).
- **`tests/integration/`**: endpoints HTTP cubriendo health, listado de
  modelos, validación y predicción, entrenamiento end-to-end. Incluye un
  test que verifica que un entrenamiento ejecutado vía API se persiste
  correctamente en la BD.

`conftest.py` configura `DATABASE_URL` a un SQLite temporal con UUID
único por sesión y llama `init_db()` para crear las tablas; las fixtures
proporcionan CSVs de muestra y un factory de DataFrames EEG sintéticos
para no depender del dataset real.

#### Linter

```bash
ruff check backend/ scripts/ tests/
ruff format --check backend/ scripts/ tests/
```

El proyecto sigue **PEP 8** para Python y las reglas estándar de **ESLint**
para JavaScript. El frontend valida también `PropTypes` en runtime.

#### Integración continua

Cada push y pull request sobre `main` dispara el workflow
`.github/workflows/ci.yml`, que ejecuta:

1. `ruff check` y `ruff format --check`.
2. `pytest` con cobertura.
3. Análisis con SonarCloud (reporta a
   <https://sonarcloud.io/project/overview?id=ajg1005_eeg-adhd-clasificacion>).

### D.3.5. Migraciones de base de datos (Alembic)

El esquema de la BD se gestiona con Alembic. Las migraciones existentes
están en `alembic/versions/`.

#### Crear una nueva migración tras cambiar el esquema ORM

```bash
alembic revision --autogenerate -m "descripcion corta"
```

#### Aplicar todas las migraciones pendientes

```bash
alembic upgrade head
```

#### Revertir la última migración

```bash
alembic downgrade -1
```

En el despliegue con Docker, el contenedor del backend ejecuta
`alembic upgrade head` automáticamente antes de arrancar Uvicorn,
garantizando que el esquema esté siempre al día.

### D.3.6. Pipeline de research (offline)

Los scripts de `scripts/` permiten regenerar los modelos servidos por
la API desde cero. No requieren la aplicación arrancada.

```bash
# Entrenamiento + CV + selección del mejor modelo ML
python scripts/train_ml.py

# Entrenamiento + CV + selección del mejor modelo DL
python scripts/train_dl.py

# Exportar el mejor modelo seleccionado (ML)
python scripts/export_model.py

# Exportar el mejor modelo seleccionado (DL)
python scripts/export_model_dl.py

# Calcular feature importance offline (cross-subject)
python scripts/feature_importance.py
```

Cada uno produce ficheros en `models/`, `results/` y `Figuras/` que la
API sirve directamente.

### D.3.7. Cómo añadir un modelo ML o DL nuevo

#### Modelo ML

1. Registrar el modelo en `scripts/ml_model_registry.py` añadiendo una
   entrada en `MODEL_SPECS` con su clase, defaults y descripción.
2. Si tiene preprocesado específico, ajustar `scripts/pipeline.py`.
3. Asegurarse de que el nombre aparece en `UI_MODEL_NAMES` si se desea
   exponer en la UI.
4. Lanzar `python scripts/train_ml.py` para regenerar el mejor modelo.

#### Modelo DL

1. Implementar la arquitectura en `scripts/tf_models.py` siguiendo el
   patrón de las existentes (función `build_*` que devuelve un
   `tf.keras.Model` compilado).
2. Registrarla en el diccionario `DL_MODEL_BUILDERS`.
3. Lanzar `python scripts/train_dl.py`.

### D.3.8. Cómo trabajar con la aplicación durante el desarrollo

Flujo recomendado para añadir una nueva funcionalidad:

1. **Crear una *issue*** en GitHub describiendo el cambio.
2. **Crear una rama** desde `main`:
   `git checkout -b feature/nombre-de-la-feature`.
3. **Implementar y testear**:
   - Si toca al backend, añadir o ajustar tests en `tests/unit/` o
     `tests/integration/`.
   - Si toca al frontend, validar manualmente en la SPA arrancada.
4. **Pasar linter** (`ruff check`, `eslint`).
5. **Commit con mensaje descriptivo** referenciando la *issue*.
6. **Abrir Pull Request** hacia `main` y esperar a que el CI pase verde.
7. **Mergear** una vez aprobado.

---

## D.4. Documentación interna del código

El proyecto adopta varias capas complementarias de documentación interna
acorde a las recomendaciones de la guía oficial del TFG de la UBU:

### D.4.1. Documentación automática de la API REST (FastAPI)

FastAPI genera automáticamente documentación interactiva a partir de los
esquemas Pydantic y las firmas tipadas de los endpoints. Está disponible
en dos formatos sin configuración adicional:

| URL                                 | Formato            | Uso recomendado                                            |
|-------------------------------------|--------------------|------------------------------------------------------------|
| `http://localhost:8000/docs`        | Swagger UI         | Explorar y probar los endpoints desde el navegador.        |
| `http://localhost:8000/redoc`       | ReDoc              | Vista limpia para lectura sin probar peticiones.            |
| `http://localhost:8000/openapi.json`| OpenAPI 3.0 JSON   | Importar la especificación a Postman, generar clientes, etc. |

Esta documentación se genera **a partir del tipado del código**, por lo
que se mantiene siempre coherente con la implementación.

### D.4.2. Tipado estático (type hints + Pydantic)

Todo el código del backend usa **type hints de Python 3.12** y modelos
**Pydantic v2** para validación de datos. Los modelos ORM de SQLAlchemy
2.0 usan el patrón `Mapped[T]`. Este tipado actúa como documentación
ejecutable: cualquier IDE moderno (VS Code, PyCharm) muestra firmas y
errores en tiempo real.

### D.4.3. Docstrings

El código incluye *docstrings* en las funciones y clases públicas más
relevantes. Se ha seguido una estrategia selectiva: documentar primero
los contratos públicos que sirven de frontera entre capas y los puntos
críticos para reproducir los experimentos.

La ampliación se ha centrado en:

- Routers de FastAPI: endpoints de salud, modelos, predicción,
  entrenamiento y experimentos.
- Servicios principales: `training_service`, `model_service`,
  `csv_service` y runners de entrenamiento.
- Persistencia: `save_experiment`, `list_experiments`,
  `get_experiment` y clases ORM `Dataset`, `Experiment` y
  `ExperimentFold`.
- Scripts principales: `train_ml.py`, `train_dl.py`,
  `export_model.py`, `export_model_dl.py` y `feature_importance.py`.

Como evidencia cuantitativa, se midieron los objetos públicos de
`backend/` y `scripts/` mediante inspección AST de Python, ignorando
helpers privados con prefijo `_`:

| Momento        | Objetos públicos documentados | Cobertura aproximada |
|----------------|-------------------------------|----------------------|
| Antes          | 13 / 152                      | 8,6 %                |
| Después        | 71 / 152                      | 46,7 %               |

El objetivo no ha sido documentar cada función trivial, sino cubrir las
piezas cuyo contrato debe entender un desarrollador que mantenga la API,
la persistencia o el pipeline experimental.

El estilo recomendado para nuevos docstrings es **Google style**:

```python
def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Busca el umbral óptimo de clasificación binaria.

    Sobre una rejilla de 61 valores entre 0.2 y 0.8, selecciona el
    umbral que maximiza la balanced accuracy en validación. En caso
    de empate, se desempata por F1-score.

    Args:
        y_true: etiquetas reales (0/1).
        y_score: scores de probabilidad continua devueltos por el modelo.

    Returns:
        El umbral óptimo a aplicar en test.
    """
```

### D.4.4. Análisis estático con SonarCloud

El proyecto está integrado con SonarCloud
(<https://sonarcloud.io/project/overview?id=ajg1005_eeg-adhd-clasificacion>).
El análisis se ejecuta automáticamente en cada push a `main` y reporta:

- **Bugs y vulnerabilidades** detectadas estáticamente.
- **Code smells** (deuda técnica).
- **Cobertura de tests** (procesada desde `pytest --cov`).
- **Duplicación de código**.

El proyecto mantiene el **Quality Gate en estado verde** como condición
de aceptación de cualquier pull request.

### D.4.5. Documentos complementarios

En la carpeta `docs/` del repositorio se incluyen documentos
complementarios:

| Documento                          | Contenido                                                            |
|------------------------------------|----------------------------------------------------------------------|
| `analisis_codigo.md`               | Análisis técnico profundo del código, decisiones y puntos a defender. |
| `licencias.md`                     | Análisis de licencias de las 30 dependencias directas.                |
| `enlaces.md`                       | Recopilación de bibliografía y datasets de referencia.                |
| `anexo_a_plan_proyecto.md`         | Plan de proyecto, sprints y estudio de viabilidad.                    |
| `anexo_b_requisitos.md`            | Especificación de requisitos y casos de uso.                          |
| `anexo_c_diseno.md`                | Diseño de datos, arquitectura y diagramas de secuencia.               |
| `anexo_d_doc_programador.md`       | Este documento.                                                       |

---

## D.5. Métricas del código

Las siguientes métricas reflejan el tamaño y la organización del
repositorio en el momento de la entrega.

| Métrica                                | Valor       |
|----------------------------------------|-------------|
| Líneas de Python en `backend/`         | ≈ 1 734     |
| Líneas de Python en `scripts/`         | ≈ 2 386     |
| Líneas de Python en `tests/`           | ≈ 772       |
| Líneas de JS/JSX en `frontend/src/`    | ≈ 1 653     |
| Líneas de CSS                          | ≈ 788       |
| Líneas en notebooks de experimentación | ≈ 307       |
| Líneas en ficheros de configuración    | ≈ 161       |
| **Total código fuente real**           | **≈ 7 800** |
| Archivos Python en `backend/`          | 28          |
| Archivos Python en `scripts/`          | 22          |
| Archivos de test                       | 16          |
| Componentes React                      | 11          |
| Endpoints HTTP expuestos               | 11          |
| Tablas en la BD                        | 3           |

Las métricas detalladas de calidad (cobertura de tests, deuda técnica,
duplicación, etc.) se mantienen actualizadas en SonarCloud.
