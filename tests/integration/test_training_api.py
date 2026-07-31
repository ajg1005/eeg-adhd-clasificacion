import json

import pandas as pd


# comprueba que /training/options expone tipos de modelo ML y DL
def test_training_options_endpoint(auth_client):
    response = auth_client.get("/training/options")

    assert response.status_code == 200
    data = response.json()
    assert "ml" in data["model_types"]
    assert "dl" in data["model_types"]


# comprueba que /training/dataset/stats analiza correctamente un CSV valido
def test_training_dataset_stats_endpoint(
    auth_client, post_csv, valid_eeg_dataset_csv_path
):
    response = post_csv(
        auth_client, valid_eeg_dataset_csv_path, "/training/dataset/stats"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["n_patients"] == 4
    assert data["class_distribution"] == {"ADHD": 2, "Control": 2}
    assert data["missing_required_columns"] == []


# comprueba que /training/dataset/stats lista las columnas que faltan
def test_training_dataset_stats_reports_missing_columns(
    auth_client, post_csv, invalid_missing_columns_csv_path
):
    response = post_csv(
        auth_client, invalid_missing_columns_csv_path, "/training/dataset/stats"
    )

    assert response.status_code == 200
    assert "Fp1" in response.json()["missing_required_columns"]


def test_training_dataset_upload_lists_saved_dataset(
    auth_client, post_csv, valid_eeg_dataset_csv_path
):
    upload_response = post_csv(
        auth_client, valid_eeg_dataset_csv_path, "/training/datasets"
    )

    assert upload_response.status_code == 200
    uploaded = upload_response.json()
    assert uploaded["filename"] == valid_eeg_dataset_csv_path.name
    assert uploaded["reusable"] is True

    list_response = auth_client.get("/training/datasets")
    assert list_response.status_code == 200
    assert any(
        dataset["id"] == uploaded["id"] for dataset in list_response.json()["datasets"]
    )

    stats_response = auth_client.get(f"/training/datasets/{uploaded['id']}/stats")
    assert stats_response.status_code == 200
    assert stats_response.json()["n_patients"] == 4


def test_saved_datasets_are_isolated_between_users(
    auth_client,
    auth_client_factory,
    post_csv,
    valid_eeg_dataset_csv_path,
):
    upload_response = post_csv(
        auth_client,
        valid_eeg_dataset_csv_path,
        "/training/datasets",
    )
    dataset_id = upload_response.json()["id"]
    other_client, _ = auth_client_factory()

    list_response = other_client.get("/training/datasets")
    stats_response = other_client.get(f"/training/datasets/{dataset_id}/stats")

    assert all(
        dataset["id"] != dataset_id for dataset in list_response.json()["datasets"]
    )
    assert stats_response.status_code == 400
    assert stats_response.json()["detail"] == "Dataset no encontrado."

    second_upload = post_csv(
        other_client,
        valid_eeg_dataset_csv_path,
        "/training/datasets",
    )
    assert second_upload.json()["id"] == dataset_id
    assert other_client.get(f"/training/datasets/{dataset_id}/stats").status_code == 200


def test_dataset_analysis_is_queued(auth_client, auth_user, monkeypatch):
    checked_access = []
    queued = []

    class TaskResult:
        id = "dataset-task-1"

    def check_access(dataset_id, user_id):
        checked_access.append((dataset_id, user_id))

    def enqueue(dataset_id, user_id):
        queued.append((dataset_id, user_id))
        return TaskResult()

    monkeypatch.setattr(
        "backend.datasets.router.ensure_saved_dataset_access",
        check_access,
    )
    monkeypatch.setattr("backend.datasets.router.analyze_dataset.delay", enqueue)

    response = auth_client.post("/training/datasets/7/analysis")

    expected_call = (7, auth_user["id"])
    assert response.status_code == 202
    assert response.json() == {"task_id": "dataset-task-1", "status": "PENDING"}
    assert checked_access == [expected_call]
    assert queued == [expected_call]


# comprueba que /training/run guarda el CSV y encola el entrenamiento
def test_training_run_queues_ml_training(
    auth_client, auth_user, eeg_dataframe_factory, monkeypatch
):
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
    response = auth_client.post(
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
    assert queued["owner_id"] == auth_user["id"]
    assert queued["model_type"] == "ml"
    assert queued["model_name"] == "random_forest"
    assert queued["eeg_params"]["epoch_size"] == 16
    assert queued["model_params"] == {"n_estimators": 5, "max_depth": 2}
    assert queued["training_params"] == {}


# comprueba que la tarea ejecuta el flujo ML completo y persiste sus resultados
def test_training_task_ml_returns_metrics_and_feature_importance(
    auth_client, auth_user, eeg_dataframe_factory
):
    from backend.datasets.service import save_training_dataset
    from backend.training.tasks import execute_training_task

    rows = eeg_dataframe_factory(samples_per_patient=32)
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    dataset = save_training_dataset(
        csv_bytes,
        "training.csv",
        user_id=auth_user["id"],
    )

    data = execute_training_task.run(
        dataset_id=dataset["id"],
        owner_id=auth_user["id"],
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

    detail_response = auth_client.get(f"/experiments/{data['experiment_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["model_name"] == "random_forest"
    assert detail["display_name"] == "Random Forest"
    assert detail["dataset"]["filename"] == "training.csv"
    assert detail["fold_results"]

    list_response = auth_client.get("/experiments?model_type=ml")
    assert list_response.status_code == 200
    assert any(
        experiment["id"] == data["experiment_id"]
        for experiment in list_response.json()["experiments"]
    )


# comprueba que /training/run rechaza un dataset con una sola clase
def test_training_run_rejects_single_class_dataset(auth_client, eeg_dataframe_factory):
    rows = eeg_dataframe_factory(
        patients=[("control_1", 0), ("control_2", 0)],
        samples_per_patient=32,
    )
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    response = auth_client.post(
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


def test_experiment_detail_returns_404_for_unknown_id(auth_client):
    response = auth_client.get("/experiments/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Experimento no encontrado."
