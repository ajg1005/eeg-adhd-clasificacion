from celery import Celery

from backend.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND


celery_app = Celery(
    "eeg_adhd",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["backend.worker.tasks", "backend.datasets.tasks"],
)

celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    enable_utc=True,
    result_expires=3600,
    result_serializer="json",
    task_serializer="json",
    task_track_started=True,
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
