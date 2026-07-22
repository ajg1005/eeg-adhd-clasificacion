from backend.training.tasks import execute_training_task


def test_execute_training_task_loads_dataset_and_runs_training(monkeypatch):
    expected = {"accuracy": 0.8}
    captured = {}

    monkeypatch.setattr(
        "backend.training.tasks.get_saved_dataset_file",
        lambda dataset_id: (b"csv-data", "dataset.csv"),
    )

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "backend.training.tasks.run_training",
        fake_run_training,
    )

    result = execute_training_task.run(
        dataset_id=7,
        model_type="ml",
        model_name="random_forest",
        eeg_params={"epoch_size": 512},
        model_params={"n_estimators": 100},
        training_params={},
    )

    assert result == expected
    assert captured == {
        "file_bytes": b"csv-data",
        "filename": "dataset.csv",
        "model_type": "ml",
        "model_name": "random_forest",
        "eeg_params": {"epoch_size": 512},
        "model_params": {"n_estimators": 100},
        "training_params": {},
    }
