import pandas as pd

from backend.experiments.repository import save_experiment
from backend.model_registry.repository import save_trained_model
from tests.conftest import requires_ml_model


def _save_training_experiment(
    eeg_dataframe_factory,
    model_name="random_forest",
    *,
    balanced_accuracy=0.79,
    f1_score=0.79,
):
    df = pd.DataFrame(eeg_dataframe_factory(samples_per_patient=16))
    file_bytes = df.to_csv(index=False).encode("utf-8")
    result = {
        "accuracy": 0.8,
        "balanced_accuracy": balanced_accuracy,
        "precision": 0.78,
        "recall": 0.81,
        "f1_score": f1_score,
        "training_time_seconds": 0.2,
        "confusion_matrix": [[3, 1], [1, 3]],
        "classification_report": {},
        "fold_results": [],
        "configuration": {
            "model_type": "ml",
            "model_name": model_name,
            "evaluation_mode": "cross_subject",
            "eeg_params": {"epoch_size": 1920, "step_size": 960},
            "model_params": {},
            "training_params": {},
        },
    }
    return save_experiment(file_bytes, "training.csv", df, result)


def _register_trained_model(
    experiment_id,
    tmp_path,
    *,
    model_name="random_forest",
    create_artifact=True,
):
    model_dir = tmp_path / f"trained-model-{experiment_id}"
    model_dir.mkdir()
    artifact_path = model_dir / "model.joblib"
    feature_columns_path = model_dir / "feature_columns.json"

    if create_artifact:
        artifact_path.write_bytes(b"fake-model")
        feature_columns_path.write_text("[]", encoding="utf-8")

    return save_trained_model(
        experiment_id,
        {
            "model_type": "ml",
            "model_name": model_name,
            "model_family": "machine_learning",
            "artifact_path": str(artifact_path),
            "feature_columns_path": str(feature_columns_path),
            "n_features": 0,
            "n_epochs_training": 8,
            "n_subjects_training": 4,
            "file_size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
            "threshold": None,
            "model_metadata": {"model_name": model_name},
            "is_selected": False,
        },
    )


# comprueba que /models lista los modelos base y los registrados tras entrenar
def test_models_endpoint_lists_static_and_registered_trained_models(
    client,
    eeg_dataframe_factory,
    tmp_path,
):
    experiment_id = _save_training_experiment(eeg_dataframe_factory)
    trained_model_id = _register_trained_model(experiment_id, tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    models = {model["model_id"]: model for model in response.json()["models"]}
    item = models[f"trained_model_{trained_model_id}"]

    assert models["ml_best"]["display_name"] == "Mejor modelo ML"
    assert models["dl_best"]["display_name"] == "Mejor modelo Deep Learning"
    assert item["display_name"] == f"Random Forest - experimento #{experiment_id}"
    assert item["model_family"] == "machine_learning"
    assert item["description"] == "Modelo entrenado desde la aplicacion"
    assert item["enabled"] is True


# comprueba que un registro sin artefacto disponible queda deshabilitado
def test_models_endpoint_marks_missing_artifact_as_disabled(
    client,
    eeg_dataframe_factory,
    tmp_path,
):
    experiment_id = _save_training_experiment(eeg_dataframe_factory, model_name="xgboost")
    trained_model_id = _register_trained_model(
        experiment_id,
        tmp_path,
        model_name="xgboost",
        create_artifact=False,
    )

    response = client.get("/models")

    assert response.status_code == 200
    models = {model["model_id"]: model for model in response.json()["models"]}
    item = models[f"trained_model_{trained_model_id}"]

    assert item["display_name"] == f"XGBoost - experimento #{experiment_id}"
    assert item["enabled"] is False


def test_best_model_endpoint_returns_highest_ranked_available_artifact(
    client,
    eeg_dataframe_factory,
    tmp_path,
):
    missing_experiment_id = _save_training_experiment(
        eeg_dataframe_factory,
        model_name="xgboost",
        balanced_accuracy=0.95,
        f1_score=0.94,
    )
    _register_trained_model(
        missing_experiment_id,
        tmp_path,
        model_name="xgboost",
        create_artifact=False,
    )

    available_experiment_id = _save_training_experiment(
        eeg_dataframe_factory,
        balanced_accuracy=0.87,
        f1_score=0.86,
    )
    trained_model_id = _register_trained_model(
        available_experiment_id,
        tmp_path,
    )

    response = client.get("/models/best")

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == f"trained_model_{trained_model_id}"
    assert data["experiment_id"] == available_experiment_id
    assert data["display_name"] == "Random Forest"
    assert data["balanced_accuracy"] == 0.87
    assert data["dataset_filename"] == "training.csv"


def test_best_model_endpoint_returns_null_without_registered_models(client, monkeypatch):
    monkeypatch.setattr(
        "backend.model_registry.repository.list_trained_models_ranked",
        lambda: [],
    )
    response = client.get("/models/best")

    assert response.status_code == 200
    assert response.json() is None


# comprueba que /model/info devuelve metadatos del modelo seleccionado
@requires_ml_model
def test_model_info_endpoint_returns_metadata(client):
    response = client.get("/model/info", params={"model_id": "ml_best"})

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "ml_best"
    assert data["channels"]


# comprueba que /model/info devuelve 404 para un modelo que no existe
def test_model_info_endpoint_rejects_unknown_model(client):
    response = client.get("/model/info", params={"model_id": "unknown"})

    assert response.status_code == 404