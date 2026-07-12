import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.config import BASE_DIR, DATASETS_DIR
from backend.constants import normalize_class_to_label
from backend.db.engine import SessionLocal
from backend.db.models import Dataset


def save_dataset(
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Guarda o reutiliza un dataset de entrenamiento identificado por hash."""
    with SessionLocal() as session:
        dataset = get_or_create_dataset(session, file_bytes, filename, dataframe)
        session.commit()
        session.refresh(dataset)
        return _dataset_to_dict(dataset)


def list_datasets(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Lista datasets guardados para poder reutilizarlos en entrenamientos."""
    with SessionLocal() as session:
        stmt = (
            select(Dataset)
            .order_by(Dataset.created_at.desc(), Dataset.id.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 200)))
        )
        return [_dataset_to_dict(dataset) for dataset in session.scalars(stmt).all()]


def load_dataset_file(dataset_id: int) -> tuple[bytes, str]:
    """Devuelve los bytes y el nombre de un CSV persistido."""
    with SessionLocal() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError("Dataset no encontrado.")
        if not dataset.storage_path:
            raise ValueError("El dataset no tiene archivo asociado para reutilizar.")

        path = _resolve_storage_path(dataset.storage_path)
        if not path.exists():
            raise ValueError("No se encuentra el archivo del dataset guardado.")

        return path.read_bytes(), dataset.filename


def get_or_create_dataset(
    session: Session,
    file_bytes: bytes,
    filename: str,
    dataframe: pd.DataFrame,
):
    """Obtiene un dataset por hash o crea su registro y archivo persistente."""
    dataset_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = session.scalar(
        select(Dataset).where(Dataset.dataset_hash == dataset_hash)
    )
    if existing is not None:
        _ensure_dataset_file(existing, file_bytes, filename)
        session.flush()
        return existing

    storage_path = _write_dataset_file(dataset_hash, file_bytes)
    dataset = Dataset(
        dataset_hash=dataset_hash,
        filename=filename or "training.csv",
        original_filename=filename or "training.csv",
        storage_path=storage_path,
        file_size_bytes=len(file_bytes),
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        n_subjects=int(dataframe["ID"].nunique()) if "ID" in dataframe.columns else 0,
        class_distribution=_class_distribution(dataframe),
        eeg_columns=[
            column
            for column in dataframe.columns
            if column not in {"Class", "ID"}
        ],
    )
    session.add(dataset)
    try:
        session.flush()
    except IntegrityError:
        # Otra peticion concurrente ya inserto el mismo dataset.
        session.rollback()
        return session.scalar(
            select(Dataset).where(Dataset.dataset_hash == dataset_hash)
        )
    return dataset


def _ensure_dataset_file(
    dataset: Dataset,
    file_bytes: bytes,
    filename: str,
) -> None:
    if dataset.storage_path:
        return

    dataset.storage_path = _write_dataset_file(dataset.dataset_hash, file_bytes)
    dataset.original_filename = (
        dataset.original_filename or filename or dataset.filename
    )
    dataset.file_size_bytes = dataset.file_size_bytes or len(file_bytes)


def _write_dataset_file(dataset_hash: str, file_bytes: bytes) -> str:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASETS_DIR / f"{dataset_hash}.csv"
    if not path.exists():
        path.write_bytes(file_bytes)

    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def _resolve_storage_path(storage_path: str) -> Path:
    path = Path(storage_path)
    return path if path.is_absolute() else BASE_DIR / path


def _dataset_to_dict(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": int(dataset.id),
        "dataset_hash": dataset.dataset_hash,
        "filename": dataset.filename,
        "original_filename": dataset.original_filename,
        "storage_path": dataset.storage_path,
        "file_size_bytes": dataset.file_size_bytes,
        "rows": int(dataset.rows),
        "columns": int(dataset.columns),
        "n_subjects": int(dataset.n_subjects),
        "class_distribution": dataset.class_distribution,
        "eeg_columns": dataset.eeg_columns,
        "created_at": dataset.created_at,
        "reusable": bool(dataset.storage_path),
    }


def _class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    if "Class" not in dataframe.columns:
        return {}
    labels = dataframe["Class"].map(normalize_class_to_label)
    return {
        str(label): int(count)
        for label, count in labels.value_counts(dropna=False).items()
    }
