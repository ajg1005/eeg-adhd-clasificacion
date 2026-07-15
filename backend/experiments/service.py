from typing import Any

import pandas as pd

from backend.experiments import repository
from backend.model_registry import catalog


def save_experiment(
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
    result: dict[str, Any],
) -> int:
    return repository.save_experiment(file_bytes, filename, dataframe, result)


def list_experiments(
    model_type: str | None = None,
    model_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    experiments = repository.list_experiments(model_type, model_name, limit, offset)
    return [_experiment_response(experiment) for experiment in experiments]


def get_experiment(experiment_id: int):
    experiment = repository.get_experiment(experiment_id)
    if experiment is None:
        return None

    response = _experiment_response(experiment)
    response.update(
        {
            "eeg_params": experiment.eeg_params,
            "model_params": experiment.model_params,
            "training_params": experiment.training_params,
            "confusion_matrix": experiment.confusion_matrix,
            "classification_report": experiment.classification_report,
            "fold_results": experiment.fold_results,
        }
    )
    return response


def _experiment_response(experiment) -> dict[str, Any]:
    return {
        "id": int(experiment.id),
        "created_at": experiment.created_at,
        "model_type": experiment.model_type,
        "model_name": experiment.model_name,
        "display_name": catalog.display_name(experiment.model_name),
        "evaluation_mode": experiment.evaluation_mode,
        "training_time_seconds": float(experiment.training_time_seconds),
        "accuracy": float(experiment.accuracy),
        "balanced_accuracy": float(experiment.balanced_accuracy),
        "precision": float(experiment.precision),
        "recall": float(experiment.recall),
        "f1_score": float(experiment.f1_score),
        "dataset": experiment.dataset,
    }