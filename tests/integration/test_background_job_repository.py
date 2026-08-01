import uuid

from backend.worker.job_repository import (
    register_background_job,
    user_can_access_background_job,
)


def test_background_job_is_only_accessible_by_its_owner(
    auth_client_factory,
):
    _, owner = auth_client_factory()
    _, other_user = auth_client_factory()
    task_id = f"task-{uuid.uuid4().hex}"

    register_background_job(
        task_id=task_id,
        owner_id=owner["id"],
        task_name="training.run",
    )

    assert user_can_access_background_job(task_id, owner["id"]) is True
    assert user_can_access_background_job(task_id, other_user["id"]) is False
    assert user_can_access_background_job("missing-task", owner["id"]) is False
