from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.responses import NOT_FOUND_RESPONSES
from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.worker.celery_app import celery_app
from backend.worker.job_service import (
    BackgroundJobNotFoundError,
    ensure_background_job_access,
)


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    response_model_exclude_none=True,
    responses=NOT_FOUND_RESPONSES,
)
def task_status(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
):
    try:
        ensure_background_job_access(task_id, int(current_user.id))
    except BackgroundJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.") from exc

    task = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task.status}

    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)

    return response
