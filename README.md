# EEG ADHD Classifier

[![CI](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)

Aplicación web para clasificar señales EEG como `ADHD` o `Control` mediante modelos de Machine Learning y Deep Learning.

## Introducción

Este proyecto desarrolla un sistema de apoyo académico para la clasificación de TDAH a partir de señales EEG. La aplicación permite cargar archivos CSV, explorar el dataset, entrenar modelos de forma interactiva y obtener predicciones usando modelos previamente entrenados.

Se han implementado dos enfoques:

- Modelos clásicos de Machine Learning (SVM RBF, Random Forest, XGBoost) basados en características temporales y espectrales por ventana EEG.
- Modelos de Deep Learning (CNN 1D, CNN-LSTM) entrenados directamente sobre ventanas crudas de señal EEG.

El backend está desarrollado con FastAPI y el frontend con React. Toda la evaluación se hace con validación cruzada cross-subject (`StratifiedGroupKFold`) para evitar fuga de información por paciente.

El sistema tiene carácter académico y no debe utilizarse como herramienta de diagnóstico clínico.

## Flujo del sistema

```text
CSV EEG -> validación de columnas y pacientes -> preprocesado -> segmentación en epochs
        -> extracción de características o ventanas crudas -> entrenamiento/evaluación
        -> exportación del modelo -> predicción por epochs -> agregación por paciente
```

En los modelos clásicos de Machine Learning se utilizan características temporales y espectrales extraídas por canal y ventana EEG. En los modelos de Deep Learning se emplean directamente las ventanas crudas de señal, con filtrado y normalización por sujeto.

## Metodología de evaluación

La evaluación se realiza a nivel de paciente mediante validación **cross-subject**. Esto significa que las ventanas de un mismo paciente no aparecen simultáneamente en entrenamiento y test, reduciendo el riesgo de fuga de información entre epochs del mismo sujeto.

Para los modelos ML se usa validación cruzada con `StratifiedGroupKFold`. Para los modelos DL se emplean folds externos por sujeto y una validación interna también separada por sujeto para ajustar el umbral de clasificación.

## Interfaz

La aplicación se organiza en cuatro pestañas:

- **Modelo**: información del mejor modelo exportado (ML o DL), métricas de CV y figuras de evaluación.
- **Dataset entrenamiento**: carga un CSV con varios pacientes, muestra estadísticas (filas, columnas, clases, canales) y permite filtrar la lista de pacientes por clase.
- **Entrenamiento**: usa el dataset cargado, configura parámetros EEG/modelo/entrenamiento y lanza un entrenamiento cross-subject con resultados detallados.
- **Experimentos**: se guarda un historico de los entrenamientos realizados,diferentes parametros e informacion del dataset.
- **Predicción**: sube el CSV de un paciente, se valida contra el modelo seleccionado y se obtiene la clasificación final junto con la distribución de epochs.

## Tecnologías utilizadas

- Python 3.12, FastAPI, Pydantic
- scikit-learn, XGBoost
- TensorFlow / Keras
- pandas, NumPy, SciPy, matplotlib
- React 19 + Vite
- Recharts
- Docker, Docker Compose
- PostgreSQL
- Alembic (migraciones de esquema)

## Estructura del proyecto

```text
backend/      API FastAPI, servicios y factories de modelos consumidos por la UI
alembic/      Migraciones de la base de datos de experimentos y datasets
frontend/    Interfaz React (Vite)
scripts/      Pipeline de investigación: entrenamiento offline, export del mejor
              modelo y scripts de análisis (comparación estadística, feature
              importance)
models/       Modelos entrenados exportados y sus metadatos
results/      Resultados de validación cruzada y configuración de los mejores
              modelos
Figuras/      Figuras generadas durante los experimentos
notebooks/    Notebooks de experimentación preliminar
tests/        Tests básicos con pytest
```

## Cómo ejecutarlo

El proyecto está pensado para ejecutarse **siempre con Docker Compose**, que es
quien levanta PostgreSQL y aplica las migraciones de Alembic antes de arrancar
la API. Ejecutar `uvicorn backend.main:app` a pelo no creará el esquema de la
base de datos.

Antes del primer arranque, copia `.env.example` a `.env` y define una
contraseña para PostgreSQL:

```bash
cp .env.example .env
# edita .env y pon tu POSTGRES_PASSWORD
```

Después, levanta el stack:

```bash
docker compose up --build
```

Backend: http://localhost:8000 · Frontend: http://localhost:5173

`docker compose` aplica `alembic upgrade head` antes de arrancar el backend, de
modo que las tablas (`datasets`, `experiments`, `experiment_folds`) quedan
creadas tras la primera ejecución.


## Scripts de investigación

- `python -m scripts.train_ml`: entrena y evalúa los modelos ML con CV cross-subject.
- `python -m scripts.train_dl`: entrena y evalúa los modelos DL con CV cross-subject.
- `python -m scripts.export_model` y `python -m scripts.export_model_dl`: exportan el modelo final seleccionado.
- `python -m scripts.feature_importance`: importancia de características con permutation_importance sobre test separado cross-subject.

## Tests

Los tests unitarios e integración comprueban validación de datos, segmentación, extracción de características, particiones cross-subject, servicios de entrenamiento y endpoints principales de la API.

```bash
pytest
```

## Limitaciones

- El sistema es un prototipo académico y no está validado para uso clínico.
- Los resultados dependen del dataset utilizado y deberían contrastarse con validación externa.
- Las predicciones se calculan por epochs y se agregan a nivel de paciente, por lo que ambas métricas deben interpretarse con cautela.
- Los modelos DL pueden ser sensibles al tamaño del dataset, al preprocesado y a la variabilidad entre sujetos.
