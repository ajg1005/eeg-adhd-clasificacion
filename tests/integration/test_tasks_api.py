from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("status", "result", "expected"),
    [
        ("PENDING", None, {"task_id": "task-123", "status": "PENDING"}),
        (
            "SUCCESS",
            {"rows": 100},
            {
                "task_id": "task-123",
                "status": "SUCCESS",
                "result": {"rows": 100},
            },
        ),
        (
            "FAILURE",
            ValueError("CSV no valido"),
            {
                "task_id": "task-123",
                "status": "FAILURE",
                "error": "CSV no valido",
            },
        ),
    ],
)
def test_task_status(client, monkeypatch, status, result, expected):
    task = SimpleNamespace(
        status=status,
        result=result,
        successful=lambda: status == "SUCCESS",
        failed=lambda: status == "FAILURE",
    )
    monkeypatch.setattr("backend.api.tasks.celery_app.AsyncResult", lambda _: task)

    response = client.get("/tasks/task-123")

    assert response.status_code == 200
    assert response.json() == expected
