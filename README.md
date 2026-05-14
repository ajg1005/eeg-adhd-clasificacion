# EEG ADHD Classifier

Aplicación web para clasificar señales EEG como `ADHD` o `Control` mediante modelos de Machine Learning y Deep Learning.

## Introducción

Este proyecto desarrolla un sistema de apoyo académico para la clasificación de TDAH a partir de señales EEG. La aplicación permite subir un archivo CSV, validar su estructura, visualizar señales EEG y obtener una predicción usando modelos previamente entrenados.

Se han implementado dos enfoques:

- Un modelo clásico de Machine Learning basado en características temporales y espectrales.
- Un modelo de Deep Learning entrenado directamente sobre ventanas de señal EEG.

El backend está desarrollado con FastAPI y el frontend con React. Esta separación permite mantener la lógica de inferencia y la interfaz de usuario desacopladas.

El sistema tiene carácter académico y no debe utilizarse como herramienta de diagnóstico clínico.

## Tecnologías utilizadas

- Python
- FastAPI
- scikit-learn
- TensorFlow / Keras
- pandas, NumPy, SciPy
- React
- Vite
- Recharts

## Estructura del proyecto

```text
backend/      API y lógica de servicio
frontend/     Interfaz web
scripts/      Entrenamiento, inferencia y procesado de señales
models/       Modelos entrenados y metadatos
results/      Resultados de validación y configuración
Figuras/      Figuras generadas durante los experimentos
notebooks/    Notebooks de experimentación
tests/        Tests y archivos CSV de ejemplo
