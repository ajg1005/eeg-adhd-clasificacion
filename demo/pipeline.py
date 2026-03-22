from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_models(random_state=42):
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced"
            )),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state,
                class_weight="balanced"
            )),
        ]),
        "gaussian_nb": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            )),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                class_weight="balanced"
            )),
        ]),
    }

    return models