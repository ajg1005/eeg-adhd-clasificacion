from typing import Any

import pandas as pd

from backend.experiments.repository import save_experiment
from backend.model_registry.repository import (
    get_trained_model_by_experiment,
    save_trained_model,
)


def _training_result() -> dict[str, Any]:
    return {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "training_time_seconds": 0.1,
        "confusion_matrix": [[1, 0], [0, 1]],
        "classification_report": {},
        "fold_results": [],
        "configuration": {
            "model_type": "ml",
            "model_name": "random_forest",
            "evaluation_mode": "test",
            "eeg_params": {},
            "model_params": {},
            "training_params": {},
        },
    }


def _artifact() -> dict[str, Any]:
    return {
        "model_type": "ml",
        "model_name": "random_forest",
        "model_family": "machine_learning",
        "artifact_path": "models/trained/1/model.joblib",
        "feature_columns_path": "models/trained/1/feature_columns.json",
        "n_features": 10,
        "n_epochs_training": 8,
        "n_subjects_training": 4,
        "file_size_bytes": 123,
        "threshold": None,
        "model_metadata": {"model_name": "random_forest"},
        "is_selected": False,
    }


def test_save_and_get_trained_model_roundtrip(client, eeg_dataframe_factory):
    df = pd.DataFrame(eeg_dataframe_factory(samples_per_patient=16))
    file_bytes = df.to_csv(index=False).encode("utf-8")
    experiment_id = save_experiment(file_bytes, "training.csv", df, _training_result())
    trained_model_id = save_trained_model(experiment_id, _artifact())

    trained_model = get_trained_model_by_experiment(experiment_id)

    assert trained_model is not None
    assert trained_model.id == trained_model_id
    assert trained_model.experiment_id == experiment_id
    assert trained_model.model_family == "machine_learning"
    assert trained_model.artifact_path == "models/trained/1/model.joblib"
    assert trained_model.model_metadata["model_name"] == "random_forest"


def test_experiments_expose_trained_model_id(client, eeg_dataframe_factory):
    """La UI construye trained_model_<id> para promocionar un experimento a
    inferencia: sin este campo no hay forma de emparejarlos."""
    df = pd.DataFrame(eeg_dataframe_factory(samples_per_patient=16))
    file_bytes = df.to_csv(index=False).encode("utf-8")
    without_artifact = save_experiment(
        file_bytes, "training.csv", df, _training_result()
    )
    with_artifact = save_experiment(file_bytes, "training.csv", df, _training_result())
    trained_model_id = save_trained_model(with_artifact, _artifact())

    listed = {
        item["id"]: item
        for item in client.get("/experiments").json()["experiments"]
    }

    assert listed[with_artifact]["trained_model_id"] == trained_model_id
    # El paso de persistir el modelo final va en try/except: puede no existir.
    assert listed[without_artifact]["trained_model_id"] is None

    detail = client.get(f"/experiments/{with_artifact}").json()

    assert detail["trained_model_id"] == trained_model_id
