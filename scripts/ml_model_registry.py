from typing import Any


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "default_params": {"max_iter": 1000, "C": 1.0, "class_weight": "balanced"},
        "parameters": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [500, 1000, 2000],
            "class_weight": ["balanced", None],
        },
    },
    "rbf_svc": {
        "display_name": "SVM RBF",
        "default_params": {"C": 10.0, "gamma": "scale", "class_weight": "balanced"},
        "parameters": {
            "C": [1.0, 10.0],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced", None],
        },
    },
    "knn": {
        "display_name": "KNN",
        "default_params": {"n_neighbors": 5, "weights": "distance"},
        "parameters": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "criterion": "entropy",
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": "balanced",
            "min_samples_leaf": 1,
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

ALL_MODEL_NAMES = tuple(MODEL_SPECS)
UI_MODEL_NAMES = ("rbf_svc", "random_forest", "xgboost")

ML_MODEL_OPTIONS = {
    name: {
        "display_name": MODEL_SPECS[name]["display_name"],
        "default_params": MODEL_SPECS[name]["default_params"],
        "parameters": MODEL_SPECS[name]["parameters"],
    }
    for name in UI_MODEL_NAMES
}


def clean_ml_params(params: dict[str, Any] | None) -> dict[str, Any]:
    clean = dict(params or {})
    for key, value in list(clean.items()):
        if value == "none":
            clean[key] = None
    return clean


def merged_ml_params(model_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Modelo ML no soportado: {model_name}")
    defaults = MODEL_SPECS[model_name]["default_params"]
    return {**defaults, **clean_ml_params(params)}
