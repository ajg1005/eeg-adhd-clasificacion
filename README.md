# EEG ADHD Classifier

Aplicacion web para clasificar senales EEG como `ADHD` o `Control` mediante modelos de Machine Learning y Deep Learning.

## Introduccion

Este proyecto desarrolla un sistema de apoyo academico para la clasificacion de TDAH a partir de senales EEG. La aplicacion permite subir un archivo CSV, validar su estructura, visualizar senales EEG y obtener una prediccion usando modelos previamente entrenados.

Se han implementado dos enfoques:

- Un modelo clasico de Machine Learning basado en caracteristicas temporales y espectrales.
- Un modelo de Deep Learning entrenado directamente sobre ventanas de senal EEG.

El backend esta desarrollado con FastAPI y el frontend con React. Esta separacion permite mantener la logica de inferencia y la interfaz de usuario desacopladas.

El sistema tiene caracter academico y no debe utilizarse como herramienta de diagnostico clinico.

## Tecnologias utilizadas

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
backend/      API y logica de servicio
frontend/     Interfaz web
scripts/      Entrenamiento, inferencia y procesado de senales
models/       Modelos entrenados y metadatos
results/      Resultados de validacion y configuracion
Figuras/      Figuras generadas durante los experimentos
notebooks/    Notebooks de experimentacion
tests/        Tests y archivos CSV de ejemplo
```
