from types import SimpleNamespace

import pytest

from backend.worker import job_service


class FakeTask:
    name = "training.run"

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.call = None

    def apply_async(self, *, args, kwargs, task_id):
        self.call = {
            "args": args,
            "kwargs": kwargs,
            "task_id": task_id,
        }
        if self.error is not None:
            raise self.error
        return SimpleNamespace(id=task_id)


def test_enqueue_background_job_registers_owner_before_dispatch(monkeypatch):
    events = []
    task = FakeTask()

    monkeypatch.setattr(
        job_service.job_repository,
        "register_background_job",
        lambda **values: events.append(("registered", values)),
    )
    monkeypatch.setattr(
        job_service.job_repository,
        "delete_background_job",
        lambda task_id, owner_id: events.append(("deleted", task_id, owner_id)),
    )

    result = job_service.enqueue_background_job(
        task,
        owner_id=7,
        args=(3,),
        kwargs={"model_name": "random_forest"},
    )

    assert events == [
        (
            "registered",
            {
                "task_id": result.id,
                "owner_id": 7,
                "task_name": "training.run",
            },
        )
    ]
    assert task.call == {
        "args": (3,),
        "kwargs": {"model_name": "random_forest"},
        "task_id": result.id,
    }


def test_enqueue_background_job_removes_registration_if_dispatch_fails(monkeypatch):
    events = []
    task = FakeTask(RuntimeError("broker unavailable"))

    monkeypatch.setattr(
        job_service.job_repository,
        "register_background_job",
        lambda **values: events.append(("registered", values)),
    )
    monkeypatch.setattr(
        job_service.job_repository,
        "delete_background_job",
        lambda task_id, owner_id: events.append(("deleted", task_id, owner_id)),
    )

    with pytest.raises(RuntimeError, match="broker unavailable"):
        job_service.enqueue_background_job(task, owner_id=9)

    registered_task_id = events[0][1]["task_id"]
    assert events == [
        (
            "registered",
            {
                "task_id": registered_task_id,
                "owner_id": 9,
                "task_name": "training.run",
            },
        ),
        ("deleted", registered_task_id, 9),
    ]
