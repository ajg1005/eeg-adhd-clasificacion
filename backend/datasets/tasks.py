from typing import Any

from backend.datasets.service import get_saved_dataset_stats
from backend.worker.celery_app import celery_app


@celery_app.task(name="datasets.analyze")
def analyze_dataset(dataset_id: int) -> dict[str, Any]:
    return get_saved_dataset_stats(dataset_id)
