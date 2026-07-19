from typing import Any

from backend.datasets.service import get_saved_dataset_file
from backend.training.service import run_training
from backend.worker.celery_app import celery_app


@celery_app.task(name="training.run")
def execute_training_task(
    dataset_id: int,
    model_type: str,
    model_name: str,
    eeg_params: dict[str, Any],
    model_params: dict[str, Any],
    training_params: dict[str, Any],
) -> dict[str, Any]:
    file_bytes, filename = get_saved_dataset_file(dataset_id)

    return run_training(
        file_bytes=file_bytes,
        filename=filename,
        model_type=model_type,
        model_name=model_name,
        eeg_params=eeg_params,
        model_params=model_params,
        training_params=training_params,
    )
