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

## Interfaz

La aplicación se organiza en cuatro pestañas:

- **Modelo**: información del mejor modelo exportado (ML o DL), métricas de CV y figuras de evaluación.
- **Dataset entrenamiento**: carga un CSV con varios pacientes, muestra estadísticas (filas, columnas, clases, canales) y permite filtrar la lista de pacientes por clase.
- **Entrenamiento**: usa el dataset cargado, configura parámetros EEG/modelo/entrenamiento y lanza un entrenamiento cross-subject con resultados detallados.
- **Predicción**: sube el CSV de un paciente, se valida contra el modelo seleccionado y se obtiene la clasificación final junto con la distribución de epochs.

## Tecnologías utilizadas

- Python 3.12, FastAPI, Pydantic
- scikit-learn, XGBoost
- TensorFlow / Keras
- pandas, NumPy, SciPy, matplotlib
- React 19 + Vite
- Recharts
- Docker, Docker Compose

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

Antes del primer arranque, copia `.env.example` a `.env` y define una contraseña
para PostgreSQL:

```bash
cp .env.example .env
# edita .env y pon tu POSTGRES_PASSWORD
```

Con Docker Compose:

```bash
docker compose up --build
```

Backend: http://localhost:8000 · Frontend: http://localhost:5173

Docker Compose levanta tambien PostgreSQL. El backend crea las tablas necesarias
al arrancar si todavia no existen.


## Scripts de investigación

- `python scripts/train_ml.py`: entrena y evalúa los modelos ML con CV cross-subject.
- `python scripts/train_dl.py`: entrena y evalúa los modelos DL con CV cross-subject.
- `python scripts/export_model.py` y `export_model_dl.py`: exportan el modelo final seleccionado.
- `python scripts/feature_importance.py`: importancia de features con permutation_importance sobre test held-out cross-subject.
