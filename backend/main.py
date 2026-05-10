from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from fastapi.staticfiles import StaticFiles


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"

sys.path.append(str(SCRIPTS_DIR))


# Imports de la pipeline
from inference import (
    load_model_artifacts,
    predict_eeg_dataframe,
    validate_eeg_dataframe,
)

FIGURES_DIR = BASE_DIR / "Figuras"

# Crear API
app = FastAPI(
    title="EEG ADHD Classifier API",
    description="API para clasificacion ADHD/Control a partir de senales EEG.",
    version="0.1.0",
)

app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


# Permitir llamadas desde React en local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Comprobar que la API esta viva
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


# Devolver informacion del modelo cargado
@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    try:
        model, feature_columns, metadata, metrics = load_model_artifacts()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model_name": metadata.get("model_name"),
        "feature_mode": metadata.get("feature_mode"),
        "sfreq": metadata.get("sfreq"),
        "epoch_size": metadata.get("epoch_size"),
        "step_size": metadata.get("step_size"),
        "channels": metadata.get("channels", []),
        "n_features": len(feature_columns),
        "metrics": metrics,
        "metadata": metadata,
    }


# Validar CSV antes de ejecutar la prediccion
@app.post("/validate")
async def validate_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Solo se admiten archivos CSV.",
        )

    try:
        _, _, metadata, _ = load_model_artifacts()

        df = pd.read_csv(file.file)
        expected_channels = metadata.get("channels", [])

        validate_eeg_dataframe(df, expected_channels)

        available_channels = [
            channel for channel in expected_channels
            if channel in df.columns
        ]

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "valid": True,
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "available_channels": available_channels,
        "expected_channels": expected_channels,
        "has_id": "ID" in df.columns,
        "has_class": "Class" in df.columns,
    }


# Ejecutar prediccion con el modelo
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Solo se admiten archivos CSV.",
        )

    try:
        df = pd.read_csv(file.file)
        result = predict_eeg_dataframe(df)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result

# Devolver muestras de un canal para visualizar la senal
@app.post("/preview")
async def preview_signal(
    file: UploadFile = File(...),
    channel: str = "Fp1",
    max_points: int = 1000,
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Solo se admiten archivos CSV.",
        )

    try:
        _, _, metadata, _ = load_model_artifacts()

        df = pd.read_csv(file.file)
        expected_channels = metadata.get("channels", [])

        if channel not in expected_channels:
            raise ValueError(f"El canal {channel} no esta entre los canales esperados.")

        if channel not in df.columns:
            raise ValueError(f"El archivo no contiene el canal {channel}.")

        n_points = min(max_points, len(df))
        signal = df[channel].iloc[:n_points].astype(float).tolist()

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "channel": channel,
        "n_points": int(n_points),
        "samples": [
            {
                "sample": index,
                "value": value,
            }
            for index, value in enumerate(signal)
        ],
    }

# Listar figuras del modelo para mostrarlas en React
@app.get("/model/figures")
def model_figures():
    figures = [
        {
            "title": "Comparacion por F1",
            "url": "/figures/cv_model_comparison_f1.png",
        },
        {
            "title": "Comparacion por balanced accuracy",
            "url": "/figures/cv_model_comparison_balanced_accuracy.png",
        },
        {
            "title": "Matriz de confusion",
            "url": "/figures/random_forest_cv_confusion_matrix.png",
        },
        {
            "title": "Curva ROC",
            "url": "/figures/random_forest_cv_roc_curve.png",
        },
    ]

    return {
        "figures": figures,
    }

