"""Catalogo unico de modelos disponibles en la app.

Fuente de verdad de la presentacion de cada arquitectura (nombre visible, tipo y
familia). Lo consumen tanto el flujo de entrenamiento (opciones de la UI) como el
de inferencia (selector y modelos entrenados), de modo que un mismo modelo se
muestra igual en las dos pantallas y solo hay un sitio donde anadir o renombrar.

Los hiperparametros y las factories siguen viviendo en scripts.ml_model_registry
(ML) y backend.modeling.dl_factory (DL); aqui solo esta la metadata de presentacion.
"""

MACHINE_LEARNING = "machine_learning"
DEEP_LEARNING = "deep_learning"

MODEL_CATALOG: dict[str, dict[str, str]] = {
    "logistic_regression": {"display_name": "Logistic Regression", "model_type": "ml", "model_family": MACHINE_LEARNING},
    "rbf_svc": {"display_name": "SVM RBF", "model_type": "ml", "model_family": MACHINE_LEARNING},
    "knn": {"display_name": "KNN", "model_type": "ml", "model_family": MACHINE_LEARNING},
    "random_forest": {"display_name": "Random Forest", "model_type": "ml", "model_family": MACHINE_LEARNING},
    "xgboost": {"display_name": "XGBoost", "model_type": "ml", "model_family": MACHINE_LEARNING},
    "cnn_1d": {"display_name": "CNN 1D", "model_type": "dl", "model_family": DEEP_LEARNING},
    "cnn_lstm": {"display_name": "CNN-LSTM", "model_type": "dl", "model_family": DEEP_LEARNING},
}


def display_name(model_name: str) -> str:
    """Nombre visible del modelo. Si no esta en el catalogo, formatea el id."""
    entry = MODEL_CATALOG.get(model_name)
    if entry is not None:
        return entry["display_name"]
    return model_name.replace("_", " ").title()


def model_family(model_name: str, default: str = MACHINE_LEARNING) -> str:
    """Familia (machine_learning / deep_learning) del modelo, con fallback."""
    return MODEL_CATALOG.get(model_name, {}).get("model_family", default)


def model_type(model_name: str, default: str = "ml") -> str:
    """Tipo (ml / dl) del modelo, con fallback."""
    return MODEL_CATALOG.get(model_name, {}).get("model_type", default)
