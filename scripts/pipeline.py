"""Factories that build the classical ML pipelines used in experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from scripts.constants import RANDOM_STATE
    from scripts.ml_model_registry import ALL_MODEL_NAMES, merged_ml_params
except ModuleNotFoundError:
    from constants import RANDOM_STATE
    from ml_model_registry import ALL_MODEL_NAMES, merged_ml_params


# Construye los modelos ML usando los defaults del registro compartido.
def create_ml_model(model_name, params=None):
    params = merged_ml_params(model_name, params)

    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=int(params.get("max_iter", 1000)),
                C=float(params.get("C", 1.0)),
                random_state=RANDOM_STATE,
                class_weight=params["class_weight"],
            )),
            
        ], memory=None,
        )


    if model_name in {"rbf_svc", "svm_rbf"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                C=float(params["C"]),
                gamma=params["gamma"],
                class_weight=params["class_weight"],
                random_state=RANDOM_STATE,
            )),
        ], memory=None,
        )

    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights=params["weights"],
            )),
        ], memory=None,
        )

    if model_name == "random_forest":
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                criterion=params["criterion"],
                max_features=params["max_features"],
                bootstrap=bool(params["bootstrap"]),
                class_weight=params["class_weight"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=int(params["min_samples_leaf"]),
            )),
        ], memory=None,
        )

    if model_name == "xgboost":
        return Pipeline([
            ("model", XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )),
        ], memory=None,
        )

    raise ValueError(f"Modelo ML no soportado: {model_name}")


def get_models():
    return {name: create_ml_model(name) for name in ALL_MODEL_NAMES}
