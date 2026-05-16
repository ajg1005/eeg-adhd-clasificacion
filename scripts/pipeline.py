from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_STATE = 42

ALL_MODEL_NAMES = [
    "logistic_regression",
    "rbf_svc",
    "knn",
    "random_forest",
    "xgboost",
]


# Fuente de verdad para construir cualquier modelo ML del proyecto.
# La UI consume un subconjunto via backend/modeling/model_factory.
def create_ml_model(model_name, params=None):
    params = dict(params or {})

    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=int(params.get("max_iter", 1000)),
                C=float(params.get("C", 1.0)),
                random_state=RANDOM_STATE,
                class_weight=params.get("class_weight", "balanced"),
            )),
        ])

    if model_name in {"rbf_svc", "svm_rbf"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                C=float(params.get("C", 10.0)),
                gamma=params.get("gamma", "scale"),
                class_weight=params.get("class_weight", "balanced"),
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 5)),
                weights=params.get("weights", "distance"),
            )),
        ])

    if model_name == "random_forest":
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=params.get("max_depth", 10),
                criterion=params.get("criterion", "entropy"),
                max_features="sqrt",
                bootstrap=True,
                class_weight=params.get("class_weight", "balanced"),
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    if model_name == "xgboost":
        return Pipeline([
            ("model", XGBClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 4)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )),
        ])

    raise ValueError(f"Modelo ML no soportado: {model_name}")


def get_models(random_state=RANDOM_STATE):
    return {name: create_ml_model(name) for name in ALL_MODEL_NAMES}
