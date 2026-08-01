from typing import Any
from uuid import uuid4

from backend.worker import job_repository


class BackgroundJobNotFoundError(Exception):
    """Indica que el trabajo no existe o pertenece a otro usuario."""


def enqueue_background_job(
    task: Any,
    owner_id: int,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
):
    """Registra el propietario y encola una tarea con el mismo identificador."""
    task_id = str(uuid4())
    job_repository.register_background_job(
        task_id=task_id,
        owner_id=owner_id,
        task_name=str(task.name),
    )

    try:
        return task.apply_async(
            args=args,
            kwargs=kwargs or {},
            task_id=task_id,
        )
    except Exception:
        job_repository.delete_background_job(task_id, owner_id)
        raise


def ensure_background_job_access(task_id: str, owner_id: int) -> None:
    """Oculta tanto trabajos inexistentes como trabajos de otro usuario."""
    if not job_repository.user_can_access_background_job(task_id, owner_id):
        raise BackgroundJobNotFoundError(task_id)
