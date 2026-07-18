from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend.worker.celery_app import celery_app


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
)
def task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task.status}

    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)

    return response
