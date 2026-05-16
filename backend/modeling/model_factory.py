from typing import Any


ML_MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "rbf_svc": {
        "display_name": "SVM RBF",
        "default_params": {"C": 10.0, "gamma": "scale", "class_weight": "balanced"},
        "parameters": {
            "C": [1.0, 10.0],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced", None],
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "criterion": "entropy",
            "class_weight": "balanced",
        },
        "parameters": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "criterion": ["entropy", "gini"],
            "class_weight": ["balanced", None],
        },
    },
    "xgboost": {
        "display_name": "XGBoost",
        "default_params": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "parameters": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
}


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    clean = dict(params or {})
    for key, value in list(clean.items()):
        if value == "none":
            clean[key] = None
    return clean


def _model_params(model_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    options = ML_MODEL_OPTIONS.get(model_name, {})
    defaults = options.get("default_params", {})
    return {**defaults, **_clean_params(params)}


# Wrapper UI: valida que el modelo esta expuesto en la UI, mezcla los defaults
# del catalogo con los params del usuario y delega en scripts.pipeline.
def create_ml_model(model_name: str, params: dict[str, Any] | None = None):
    from scripts.pipeline import create_ml_model as _build_ml_model

    if model_name not in ML_MODEL_OPTIONS:
        raise ValueError(f"Modelo ML no disponible en la UI: {model_name}")

    merged = _model_params(model_name, params)
    return _build_ml_model(model_name, merged)
