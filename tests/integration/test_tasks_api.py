from types import SimpleNamespace
import uuid

import pytest

from backend.worker.job_repository import register_background_job


@pytest.mark.parametrize(
    ("status", "result", "expected"),
    [
        ("PENDING", None, {"status": "PENDING"}),
        (
            "SUCCESS",
            {"rows": 100},
            {
                "status": "SUCCESS",
                "result": {"rows": 100},
            },
        ),
        (
            "FAILURE",
            ValueError("CSV no valido"),
            {
                "status": "FAILURE",
                "error": "CSV no valido",
            },
        ),
    ],
)
def test_task_status(
    auth_client,
    auth_user,
    monkeypatch,
    status,
    result,
    expected,
):
    task_id = f"task-{uuid.uuid4().hex}"
    register_background_job(
        task_id=task_id,
        owner_id=auth_user["id"],
        task_name="test.task",
    )
    task = SimpleNamespace(
        status=status,
        result=result,
        successful=lambda: status == "SUCCESS",
        failed=lambda: status == "FAILURE",
    )
    monkeypatch.setattr("backend.api.tasks.celery_app.AsyncResult", lambda _: task)

    response = auth_client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json() == {"task_id": task_id, **expected}


def test_task_status_hides_foreign_and_unknown_jobs(
    auth_client_factory,
    monkeypatch,
):
    _, owner = auth_client_factory()
    other_client, _ = auth_client_factory()
    task_id = f"task-{uuid.uuid4().hex}"
    register_background_job(
        task_id=task_id,
        owner_id=owner["id"],
        task_name="training.run",
    )
    monkeypatch.setattr(
        "backend.api.tasks.celery_app.AsyncResult",
        lambda _: pytest.fail("No se debe consultar Redis sin acceso al trabajo"),
    )

    foreign_response = other_client.get(f"/tasks/{task_id}")
    missing_response = other_client.get(f"/tasks/missing-{uuid.uuid4().hex}")

    assert foreign_response.status_code == 404
    assert foreign_response.json()["detail"] == "Trabajo no encontrado."
    assert missing_response.status_code == 404
    assert missing_response.json()["detail"] == "Trabajo no encontrado."
