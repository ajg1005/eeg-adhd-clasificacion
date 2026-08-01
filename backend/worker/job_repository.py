from sqlalchemy import select

from backend.db.engine import SessionLocal
from backend.db.models import BackgroundJob


def register_background_job(
    task_id: str,
    owner_id: int,
    task_name: str,
) -> None:
    """Registra quien puede consultar un trabajo de Celery."""
    with SessionLocal() as session:
        session.add(
            BackgroundJob(
                id=task_id,
                owner_id=owner_id,
                task_name=task_name,
            )
        )
        session.commit()


def user_can_access_background_job(task_id: str, owner_id: int) -> bool:
    """Indica si el trabajo pertenece al usuario autenticado."""
    with SessionLocal() as session:
        query = select(BackgroundJob.id).where(
            BackgroundJob.id == task_id,
            BackgroundJob.owner_id == owner_id,
        )
        return session.scalar(query) is not None
