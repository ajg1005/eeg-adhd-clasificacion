from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only, selectinload

from backend.constants import normalize_class_to_label
from backend.db.engine import SessionLocal
from backend.db.models import DatasetRecord, ExperimentFoldRecord, ExperimentRecord


def save_experiment(
    *,
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
    result: dict[str, Any],
) -> int:
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
    *,
    model_type: str | None = None,
    model_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ExperimentRecord]:
    with SessionLocal() as session:
        stmt = (
            select(ExperimentRecord)
            .options(selectinload(ExperimentRecord.dataset).options(_dataset_summary_columns()))
            .order_by(ExperimentRecord.created_at.desc(), ExperimentRecord.id.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 200)))
        )
        if model_type:
            stmt = stmt.where(ExperimentRecord.model_type == model_type)
        if model_name:
            stmt = stmt.where(ExperimentRecord.model_name == model_name)

        return list(session.scalars(stmt).all())


def get_experiment(experiment_id: int) -> ExperimentRecord | None:
    with SessionLocal() as session:
        return session.get(
            ExperimentRecord,
            experiment_id,
            options=[
                selectinload(ExperimentRecord.dataset).options(_dataset_summary_columns()),
                selectinload(ExperimentRecord.folds),
            ],
        )


def _get_or_create_dataset(
    session: Session,
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
) -> DatasetRecord:
    dataset_hash = hashlib.sha256(file_bytes).hexdigest()
    dataset = session.scalar(
        select(DatasetRecord).where(DatasetRecord.dataset_hash == dataset_hash)
    )
    if dataset is not None:
        return dataset

    dataset = DatasetRecord(
        dataset_hash=dataset_hash,
        filename=filename or "training.csv",
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        n_subjects=int(dataframe["ID"].nunique()) if "ID" in dataframe.columns else 0,
        class_distribution=_class_distribution(dataframe),
        eeg_columns=[column for column in dataframe.columns if column not in {"Class", "ID"}],
    )
    session.add(dataset)
    session.flush()
    return dataset


def _experiment_from_result(dataset_id: int, result: dict[str, Any]) -> ExperimentRecord:
    configuration = result.get("configuration", {})
    return ExperimentRecord(
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


def _fold_from_result(experiment_id: int, fold: dict[str, Any]) -> ExperimentFoldRecord:
    return ExperimentFoldRecord(
        experiment_id=experiment_id,
        fold=int(fold.get("fold", 0)),
        accuracy=float(fold.get("accuracy", 0.0)),
        balanced_accuracy=float(fold.get("balanced_accuracy", 0.0)),
        precision=float(fold.get("precision", 0.0)),
        recall=float(fold.get("recall", 0.0)),
        f1_score=float(fold.get("f1_score", 0.0)),
        n_train_subjects=_optional_int(fold.get("n_train_subjects")),
        n_val_subjects=_optional_int(fold.get("n_val_subjects")),
        n_test_subjects=_optional_int(fold.get("n_test_subjects")),
        best_threshold=_optional_float(fold.get("best_threshold")),
    )


def _dataset_summary_columns():
    return load_only(
        DatasetRecord.id,
        DatasetRecord.dataset_hash,
        DatasetRecord.filename,
        DatasetRecord.rows,
        DatasetRecord.columns,
        DatasetRecord.n_subjects,
        DatasetRecord.class_distribution,
        DatasetRecord.eeg_columns,
        DatasetRecord.created_at,
    )


def _class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    if "Class" not in dataframe.columns:
        return {}
    labels = dataframe["Class"].map(normalize_class_to_label)
    return {str(label): int(count) for label, count in labels.value_counts(dropna=False).items()}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
