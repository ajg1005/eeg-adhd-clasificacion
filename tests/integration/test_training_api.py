import json

import pandas as pd


# Helper que envia un CSV al endpoint via TestClient
def _post_csv(client, path, url, data=None):
    with path.open("rb") as csv_file:
        return client.post(
            url,
            data=data or {},
            files={"file": (path.name, csv_file, "text/csv")},
        )


# comprueba que /training/options expone tipos de modelo ML y DL
def test_training_options_endpoint(client):
    response = client.get("/training/options")

    assert response.status_code == 200
    data = response.json()
    assert "ml" in data["model_types"]
    assert "dl" in data["model_types"]


# comprueba que /training/dataset/stats analiza correctamente un CSV valido
def test_training_dataset_stats_endpoint(client, valid_eeg_dataset_csv_path):
    response = _post_csv(client, valid_eeg_dataset_csv_path, "/training/dataset/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["n_patients"] == 4
    assert data["class_distribution"] == {"ADHD": 2, "Control": 2}
    assert data["missing_required_columns"] == []


# comprueba que /training/dataset/stats lista las columnas que faltan
def test_training_dataset_stats_reports_missing_columns(client, invalid_missing_columns_csv_path):
    response = _post_csv(client, invalid_missing_columns_csv_path, "/training/dataset/stats")

    assert response.status_code == 200
    assert "Fp1" in response.json()["missing_required_columns"]


# comprueba que /training/run entrena un modelo ML y devuelve metricas + importancia
def test_training_run_ml_returns_metrics_and_feature_importance(client, eeg_dataframe_factory):
    rows = eeg_dataframe_factory(samples_per_patient=32)
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    response = client.post(
        "/training/run",
        data={
            "model_type": "ml",
            "model_name": "random_forest",
            "eeg_params": json.dumps(
                {
                    "epoch_size": 16,
                    "step_size": 16,
                    "feature_mode": "temporal",
                    "use_filtering": False,
                }
            ),
            "model_params": json.dumps({"n_estimators": 5, "max_depth": 2}),
            "training_params": "{}",
        },
        files={"file": ("training.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["accuracy"] <= 1.0
    assert data["patient_results"]
    assert data["feature_importance"]["method"] == "permutation_importance"
    assert data["feature_importance"]["top_features"]


# comprueba que /training/run rechaza un dataset con una sola clase
def test_training_run_rejects_single_class_dataset(client, eeg_dataframe_factory):
    rows = eeg_dataframe_factory(
        patients=[("control_1", 0), ("control_2", 0)],
        samples_per_patient=32,
    )
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    response = client.post(
        "/training/run",
        data={
            "model_type": "ml",
            "model_name": "random_forest",
            "eeg_params": json.dumps({"epoch_size": 16, "step_size": 16}),
            "model_params": "{}",
            "training_params": "{}",
        },
        files={"file": ("single_class.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert "Control y TDAH" in response.json()["detail"]
