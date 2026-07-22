import json

import pandas as pd


# comprueba que /training/options expone tipos de modelo ML y DL
def test_training_options_endpoint(client):
    response = client.get("/training/options")

    assert response.status_code == 200
    data = response.json()
    assert "ml" in data["model_types"]
    assert "dl" in data["model_types"]


# comprueba que /training/dataset/stats analiza correctamente un CSV valido
def test_training_dataset_stats_endpoint(client, post_csv, valid_eeg_dataset_csv_path):
    response = post_csv(client, valid_eeg_dataset_csv_path, "/training/dataset/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["n_patients"] == 4
    assert data["class_distribution"] == {"ADHD": 2, "Control": 2}
    assert data["missing_required_columns"] == []


# comprueba que /training/dataset/stats lista las columnas que faltan
def test_training_dataset_stats_reports_missing_columns(
    client, post_csv, invalid_missing_columns_csv_path
):
    response = post_csv(
        client, invalid_missing_columns_csv_path, "/training/dataset/stats"
    )

    assert response.status_code == 200
    assert "Fp1" in response.json()["missing_required_columns"]


def test_training_dataset_upload_lists_saved_dataset(
    client, post_csv, valid_eeg_dataset_csv_path
):
    upload_response = post_csv(client, valid_eeg_dataset_csv_path, "/training/datasets")

    assert upload_response.status_code == 200
    uploaded = upload_response.json()
    assert uploaded["filename"] == valid_eeg_dataset_csv_path.name
    assert uploaded["reusable"] is True

    list_response = client.get("/training/datasets")
    assert list_response.status_code == 200
    assert any(
        dataset["id"] == uploaded["id"] for dataset in list_response.json()["datasets"]
    )

    stats_response = client.get(f"/training/datasets/{uploaded['id']}/stats")
    assert stats_response.status_code == 200
    assert stats_response.json()["n_patients"] == 4


def test_dataset_analysis_is_queued(client, monkeypatch):
    queued_dataset_ids = []

    class TaskResult:
        id = "dataset-task-1"

    def enqueue(dataset_id):
        queued_dataset_ids.append(dataset_id)
        return TaskResult()

    monkeypatch.setattr("backend.datasets.router.analyze_dataset.delay", enqueue)

    response = client.post("/training/datasets/7/analysis")

    assert response.status_code == 202
    assert response.json() == {"task_id": "dataset-task-1", "status": "PENDING"}
    assert queued_dataset_ids == [7]


# comprueba que /training/run guarda el CSV y encola el entrenamiento
def test_training_run_queues_ml_training(client, eeg_dataframe_factory, monkeypatch):
    queued = {}

    class TaskResult:
        id = "training-task-1"

    def enqueue(**kwargs):
        queued.update(kwargs)
        return TaskResult()

    monkeypatch.setattr(
        "backend.training.router.execute_training_task.delay",
        enqueue,
    )

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

    assert response.status_code == 202
    assert response.json() == {
        "task_id": "training-task-1",
        "status": "PENDING",
    }
    assert queued["dataset_id"] > 0
    assert queued["model_type"] == "ml"
    assert queued["model_name"] == "random_forest"
    assert queued["eeg_params"]["epoch_size"] == 16
    assert queued["model_params"] == {"n_estimators": 5, "max_depth": 2}
    assert queued["training_params"] == {}


# comprueba que la tarea ejecuta el flujo ML completo y persiste sus resultados
def test_training_task_ml_returns_metrics_and_feature_importance(
    client, eeg_dataframe_factory
):
    from backend.datasets.service import save_training_dataset
    from backend.training.tasks import execute_training_task

    rows = eeg_dataframe_factory(samples_per_patient=32)
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    dataset = save_training_dataset(csv_bytes, "training.csv")

    data = execute_training_task.run(
        dataset_id=dataset["id"],
        model_type="ml",
        model_name="random_forest",
        eeg_params={
            "epoch_size": 16,
            "step_size": 16,
            "feature_mode": "temporal",
            "use_filtering": False,
        },
        model_params={"n_estimators": 5, "max_depth": 2},
        training_params={},
    )

    assert 0.0 <= data["accuracy"] <= 1.0
    assert data["experiment_id"] > 0
    assert data["model_saved"] is True
    assert data["trained_model_id"] > 0
    assert data["patient_results"]
    assert data["feature_importance"]["method"] == "permutation_importance"
    assert data["feature_importance"]["top_features"]

    detail_response = client.get(f"/experiments/{data['experiment_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["model_name"] == "random_forest"
    assert detail["display_name"] == "Random Forest"
    assert detail["dataset"]["filename"] == "training.csv"
    assert detail["fold_results"]

    list_response = client.get("/experiments?model_type=ml")
    assert list_response.status_code == 200
    assert any(
        experiment["id"] == data["experiment_id"]
        for experiment in list_response.json()["experiments"]
    )


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


def test_experiment_detail_returns_404_for_unknown_id(client):
    response = client.get("/experiments/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Experimento no encontrado."
