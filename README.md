# EEG ADHD Classifier

[![CI](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ajg1005/eeg-adhd-clasificacion/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ajg1005_eeg-adhd-clasificacion&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ajg1005_eeg-adhd-clasificacion)

Aplicacion web para clasificar senales EEG como `ADHD` o `Control` mediante modelos de Machine Learning y Deep Learning.

## Introduccion

Este proyecto desarrolla un sistema de apoyo academico para la clasificacion de TDAH a partir de senales EEG. La aplicacion permite cargar archivos CSV, explorar el dataset, entrenar modelos de forma interactiva y obtener predicciones usando modelos previamente entrenados.

Se han implementado dos enfoques:

- Modelos clasicos de Machine Learning (SVM RBF, Random Forest, XGBoost) basados en caracteristicas temporales y espectrales por ventana EEG.
- Modelos de Deep Learning (CNN 1D, CNN-LSTM) entrenados directamente sobre ventanas crudas de senal EEG.

El backend esta desarrollado con FastAPI y el frontend con React. Toda la evaluacion se hace con validacion cruzada cross-subject (`StratifiedGroupKFold`) para evitar leakage por paciente.

El sistema tiene caracter academico y no debe utilizarse como herramienta de diagnostico clinico.

## Interfaz

La aplicacion se organiza en cuatro pestanas:

- **Modelo**: informacion del mejor modelo exportado (ML o DL), metricas de CV y figuras de evaluacion.
- **Dataset entrenamiento**: carga un CSV con varios pacientes, muestra estadisticas (filas, columnas, clases, canales) y permite filtrar la lista de pacientes por clase.
- **Entrenamiento**: usa el dataset cargado, configura parametros EEG/modelo/entrenamiento y lanza un entrenamiento cross-subject con resultados detallados.
- **Prediccion**: sube el CSV de un paciente, se valida contra el modelo seleccionado y se obtiene la clasificacion final junto con la distribucion de epochs.

## Tecnologias utilizadas

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
frontend/    Interfaz React (Vite)
scripts/      Pipeline de investigacion: entrenamiento offline, export del mejor
              modelo y scripts de analisis (comparacion estadistica, feature
              importance)
models/       Modelos entrenados exportados y sus metadatos
results/      Resultados de validacion cruzada y configuracion de los mejores
              modelos
Figuras/      Figuras generadas durante los experimentos
notebooks/    Notebooks de experimentacion preliminar
tests/        Tests basicos con pytest
```

## Como ejecutarlo

Con Docker Compose:

```bash
docker compose up --build
```

Backend: http://localhost:8000 · Frontend: http://localhost:5173


## Scripts de investigacion

- `python scripts/train_ml.py`: entrena y evalua los modelos ML con CV cross-subject.
- `python scripts/train_dl.py`: entrena y evalua los modelos DL con CV cross-subject.
- `python scripts/export_model.py` y `export_model_dl.py`: exportan el modelo final seleccionado.
- `python scripts/feature_importance.py`: importancia de features con permutation_importance sobre test held-out cross-subject.
