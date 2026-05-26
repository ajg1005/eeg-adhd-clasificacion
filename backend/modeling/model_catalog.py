from typing import Any


MAX_EPOCHS_DESCRIPTION = "Numero maximo de epocas."
BATCH_SIZE_DESCRIPTION = "Muestras por batch."
LEARNING_RATE_DESCRIPTION = "Tasa de aprendizaje."
DROPOUT_DESCRIPTION = "Regularizacion."


# Catalogo de modelos candidatos y parametros habituales de entrenamiento
MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "model_id": "random_forest",
        "display_name": "Random Forest",
        "model_family": "machine_learning",
        "description": "Ensemble de arboles robusto para features temporales y espectrales.",
        "use_case": "Modelo clasico interpretable y estable como linea base fuerte.",
        "common_parameters": [
            {"name": "n_estimators", "default": "100", "description": "Numero de arboles."},
            {"name": "max_depth", "default": "10", "description": "Profundidad maxima de cada arbol."},
            {"name": "criterion", "default": "entropy", "description": "Criterio de division de los arboles."},
            {"name": "class_weight", "default": "balanced", "description": "Compensacion si hay clases desbalanceadas."},
        ],
    },
    {
        "model_id": "rbf_svc",
        "display_name": "SVC RBF",
        "model_family": "machine_learning",
        "description": "Clasificador SVM con kernel radial para fronteras no lineales.",
        "use_case": "Comparar rendimiento con un modelo clasico no lineal.",
        "common_parameters": [
            {"name": "C", "default": "10", "description": "Regularizacion del margen."},
            {"name": "gamma", "default": "scale", "description": "Influencia del kernel RBF."},
            {"name": "class_weight", "default": "balanced", "description": "Peso de clases."},
            {"name": "probability", "default": "True", "description": "Permite obtener probabilidades."},
        ],
    },
    {
        "model_id": "xgboost",
        "display_name": "XGBoost",
        "model_family": "machine_learning",
        "description": "Boosting de arboles para capturar relaciones complejas entre features.",
        "use_case": "Modelo tabular potente para comparar con Random Forest y SVC.",
        "common_parameters": [
            {"name": "n_estimators", "default": "200", "description": "Numero de arboles secuenciales."},
            {"name": "max_depth", "default": "3-5", "description": "Complejidad de cada arbol."},
            {"name": "learning_rate", "default": "0.05-0.1", "description": "Paso de aprendizaje."},
            {"name": "subsample", "default": "0.8", "description": "Muestreo de filas para reducir sobreajuste."},
            {"name": "colsample_bytree", "default": "0.8", "description": "Muestreo de columnas por arbol."},
        ],
    },
    {
        "model_id": "cnn_1d",
        "display_name": "CNN 1D",
        "model_family": "deep_learning",
        "description": "Red convolucional 1D aplicada directamente a epochs EEG.",
        "use_case": "Detectar patrones locales en la senal temporal multicanal.",
        "common_parameters": [
            {"name": "epochs", "default": "40", "description": MAX_EPOCHS_DESCRIPTION},
            {"name": "batch_size", "default": "32", "description": BATCH_SIZE_DESCRIPTION},
            {"name": "learning_rate", "default": "0.0003", "description": "Tasa de aprendizaje Adam."},
            {"name": "dropout", "default": "0.4", "description": "Regularizacion de capas densas/convolucionales."},
            {"name": "patience", "default": "4", "description": "Early stopping."},
        ],
    },
    {
        "model_id": "cnn_lstm",
        "display_name": "CNN-LSTM",
        "model_family": "deep_learning",
        "description": "Combina convoluciones con memoria recurrente para modelar secuencia temporal.",
        "use_case": "Capturar patrones locales y dependencias temporales de la epoch.",
        "common_parameters": [
            {"name": "epochs", "default": "40", "description": MAX_EPOCHS_DESCRIPTION},
            {"name": "batch_size", "default": "32", "description": BATCH_SIZE_DESCRIPTION},
            {"name": "learning_rate", "default": "0.0003", "description": LEARNING_RATE_DESCRIPTION},
            {"name": "dropout", "default": "0.4", "description": DROPOUT_DESCRIPTION},
            {"name": "lstm_units", "default": "32", "description": "Unidades recurrentes."},
        ],
    },
]


# Separar modelos por familia para que el frontend pueda pintarlos agrupados
def get_model_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        "machine_learning": [
            model for model in MODEL_CATALOG if model["model_family"] == "machine_learning"
        ],
        "deep_learning": [
            model for model in MODEL_CATALOG if model["model_family"] == "deep_learning"
        ],
    }
