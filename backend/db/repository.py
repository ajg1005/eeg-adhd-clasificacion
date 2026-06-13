from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from backend.constants import normalize_class_to_label
from backend.db.engine import SessionLocal
from backend.db.models import Dataset, Experiment, ExperimentFold


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
        dataset = _get_or_create_dataset(session, file_bytes, filename, dataframe)
        experiment = _experiment_from_result(dataset.id, result)
        session.add(experiment)
        session.flush()

        session.add_all(
            _fold_from_result(experiment.id, fold)
            for fold in result.get("fold_results", [])
        )
        session.commit()
        return int(experiment.id)


def list_experiments(
    model_type: str | None = None,
    model_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Devuelve los experimentos guardados del mas reciente al mas antiguo.

    Acepta filtros opcionales por tipo de modelo (ml/dl) y por nombre. La
    pestaña de Experimentos usa esto para paginar y filtrar. Eager-carga el
    dataset asociado para no provocar N+1 queries al pintar la tabla.
    """
    with SessionLocal() as session:
        stmt = (
            select(Experiment)
            .options(selectinload(Experiment.dataset))
            .order_by(Experiment.created_at.desc(), Experiment.id.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 200)))
        )
        if model_type:
            stmt = stmt.where(Experiment.model_type == model_type)
        if model_name:
            stmt = stmt.where(Experiment.model_name == model_name)

        return list(session.scalars(stmt).all())


def get_experiment(experiment_id: int):
    """Devuelve un experimento concreto con su dataset y todos sus folds.

    Eager-carga las relaciones (dataset y fold_results) para que la respuesta
    del endpoint de detalle no haga consultas adicionales. Si el ID no existe
    devuelve None y el router lo traduce a 404.
    """
    with SessionLocal() as session:
        return session.get(
            Experiment,
            experiment_id,
            options=[
                selectinload(Experiment.dataset),
                selectinload(Experiment.fold_results),
            ],
        )


def _get_or_create_dataset(
    session: Session,
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
):
    dataset_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = session.scalar(
        select(Dataset).where(Dataset.dataset_hash == dataset_hash)
    )
    if existing is not None:
        return existing

    dataset = Dataset(
        dataset_hash=dataset_hash,
        filename=filename or "training.csv",
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        n_subjects=int(dataframe["ID"].nunique()) if "ID" in dataframe.columns else 0,
        class_distribution=_class_distribution(dataframe),
        eeg_columns=[column for column in dataframe.columns if column not in {"Class", "ID"}],
    )
    session.add(dataset)
    try:
        session.flush()
    except IntegrityError:
        # Otra peticion concurrente ya insertó el mismo dataset (mismo hash).
        # Hacemos rollback del savepoint implicito y devolvemos el ganador.
        session.rollback()
        return session.scalar(
            select(Dataset).where(Dataset.dataset_hash == dataset_hash)
        )
    return dataset


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
            int(fold["n_train_subjects"]) if fold.get("n_train_subjects") is not None else None
        ),
        n_val_subjects=(
            int(fold["n_val_subjects"]) if fold.get("n_val_subjects") is not None else None
        ),
        n_test_subjects=(
            int(fold["n_test_subjects"]) if fold.get("n_test_subjects") is not None else None
        ),
        best_threshold=(
            float(fold["best_threshold"]) if fold.get("best_threshold") is not None else None
        ),
    )


def _class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    if "Class" not in dataframe.columns:
        return {}
    labels = dataframe["Class"].map(normalize_class_to_label)
    return {str(label): int(count) for label, count in labels.value_counts(dropna=False).items()}
