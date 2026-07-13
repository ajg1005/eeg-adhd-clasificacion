from typing import Any

import pandas as pd

from backend.experiments import repository


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
    return repository.list_experiments(model_type, model_name, limit, offset)


def get_experiment(experiment_id: int):
    return repository.get_experiment(experiment_id)