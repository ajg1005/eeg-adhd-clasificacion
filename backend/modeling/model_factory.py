from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


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


# Crear el estimador ML elegido por el usuario desde la vista de entrenamiento.
def create_ml_model(model_name: str, params: dict[str, Any] | None = None):
    params = _model_params(model_name, params)

    if model_name in {"rbf_svc", "svm_rbf"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        C=float(params.get("C", 10.0)),
                        gamma=params.get("gamma", "scale"),
                        class_weight=params.get("class_weight", "balanced"),
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params.get("n_estimators", 100)),
                        max_depth=params.get("max_depth", 10),
                        criterion=params.get("criterion", "entropy"),
                        max_features="sqrt",
                        bootstrap=True,
                        class_weight=params.get("class_weight", "balanced"),
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        )

    if model_name == "xgboost":
        return Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        n_estimators=int(params.get("n_estimators", 200)),
                        max_depth=int(params.get("max_depth", 4)),
                        learning_rate=float(params.get("learning_rate", 0.05)),
                        subsample=float(params.get("subsample", 0.8)),
                        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                    ),
                )
            ]
        )

    raise ValueError(f"Modelo ML no soportado: {model_name}")

