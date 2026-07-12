from typing import Any

import pandas as pd

from backend.datasets import repository as dataset_repository
from backend.db.engine import SessionLocal
from backend.db.models import Experiment, ExperimentFold


def save_experiment(
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
    result: dict[str, Any],
):
    """Guarda en BD un experimento completo con sus folds.

    Si el CSV ya existia (mismo hash SHA-256) reutiliza la fila de Dataset
    en vez de duplicarla. Devuelve el ID del experimento recien creado para
    que el endpoint lo pueda incluir en la respuesta al frontend.
    """
    with SessionLocal() as session:
        dataset = dataset_repository.get_or_create_dataset(
            session,
            file_bytes,
            filename,
            dataframe,
        )
        experiment = _experiment_from_result(dataset.id, result)
        session.add(experiment)
        session.flush()

        session.add_all(
            _fold_from_result(experiment.id, fold)
            for fold in result.get("fold_results", [])
        )
        session.commit()
        return int(experiment.id)


def _experiment_from_result(dataset_id: int, result: dict[str, Any]):
    configuration = result.get("configuration", {})
    return Experiment(
        dataset_id=dataset_id,
        model_type=str(configuration.get("model_type", "")),
        model_name=str(configuration.get("model_name", "")),
        evaluation_mode=str(configuration.get("evaluation_mode", "")),
        training_time_seconds=float(result.get("training_time_seconds", 0.0)),
        accuracy=float(result.get("accuracy", 0.0)),
        balanced_accuracy=float(result.get("balanced_accuracy", 0.0)),
        precision=float(result.get("precision", 0.0)),
        recall=float(result.get("recall", 0.0)),
        f1_score=float(result.get("f1_score", 0.0)),
        eeg_params=configuration.get("eeg_params", {}),
        model_params=configuration.get("model_params", {}),
        training_params=configuration.get("training_params", {}),
        confusion_matrix=result.get("confusion_matrix", []),
        classification_report=result.get("classification_report", {}),
    )


def _fold_from_result(experiment_id: int, fold: dict[str, Any]):
    return ExperimentFold(
        experiment_id=experiment_id,
        fold=int(fold.get("fold", 0)),
        accuracy=float(fold.get("accuracy", 0.0)),
        balanced_accuracy=float(fold.get("balanced_accuracy", 0.0)),
        precision=float(fold.get("precision", 0.0)),
        recall=float(fold.get("recall", 0.0)),
        f1_score=float(fold.get("f1_score", 0.0)),
        n_train_subjects=(
            int(fold["n_train_subjects"])
            if fold.get("n_train_subjects") is not None
            else None
        ),
        n_val_subjects=(
            int(fold["n_val_subjects"])
            if fold.get("n_val_subjects") is not None
            else None
        ),
        n_test_subjects=(
            int(fold["n_test_subjects"])
            if fold.get("n_test_subjects") is not None
            else None
        ),
        best_threshold=(
            float(fold["best_threshold"])
            if fold.get("best_threshold") is not None
            else None
        ),
    )
