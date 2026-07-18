from backend.datasets.tasks import analyze_dataset


def test_analyze_dataset_task_returns_saved_stats(monkeypatch):
    expected = {"rows": 100, "n_patients": 4}

    monkeypatch.setattr(
        "backend.datasets.tasks.get_saved_dataset_stats",
        lambda dataset_id: expected,
    )

    assert analyze_dataset.run(7) == expected
