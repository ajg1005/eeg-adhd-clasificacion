from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from backend.datasets import repository
from scripts.constants import (
    REQUIRED_COLUMNS,
    REQUIRED_EEG_COLUMNS,
    normalize_class_to_label,
)
from scripts.validators import validate_training_dataframe


def read_csv(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("El archivo CSV esta vacio.")
    return pd.read_csv(BytesIO(file_bytes))


def get_dataset_stats(file_bytes: bytes, preview_rows: int = 5) -> dict[str, Any]:
    df = read_csv(file_bytes)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "n_patients": int(df["ID"].nunique()) if "ID" in df.columns else 0,
        "class_distribution": _class_distribution(df),
        "patients": _patient_rows(df),
        "eeg_columns": [column for column in REQUIRED_EEG_COLUMNS if column in df.columns],
        "missing_required_columns": _missing_required_columns(df),
        "preview": df.head(preview_rows).replace({np.nan: None}).to_dict(orient="records"),
    }


def get_saved_datasets(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return repository.list_datasets(user_id, limit=limit, offset=offset)


def save_training_dataset(
    file_bytes: bytes,
    filename: str,
    user_id: int,
) -> dict[str, Any]:
    df = read_csv(file_bytes)
    validate_training_dataframe(df)
    return repository.save_dataset(
        file_bytes=file_bytes,
        filename=filename,
        dataframe=df,
        user_id=user_id,
    )


def get_saved_dataset_stats(
    dataset_id: int,
    user_id: int,
    preview_rows: int = 5,
) -> dict[str, Any]:
    file_bytes, _ = repository.load_dataset_file(dataset_id, user_id)
    return get_dataset_stats(file_bytes, preview_rows=preview_rows)


def get_saved_dataset_file(dataset_id: int, user_id: int) -> tuple[bytes, str]:
    return repository.load_dataset_file(dataset_id, user_id)


def ensure_saved_dataset_access(dataset_id: int, user_id: int) -> None:
    if not repository.user_can_access_dataset(dataset_id, user_id):
        raise ValueError("Dataset no encontrado.")


def _missing_required_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def _class_distribution(df: pd.DataFrame) -> dict[str, int]:
    if "Class" not in df.columns:
        return {}

    labels = df["Class"].map(normalize_class_to_label)
    return {str(label): int(count) for label, count in labels.value_counts(dropna=False).items()}


def _patient_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "ID" not in df.columns:
        return []

    rows = []
    for patient_id, patient_df in df.groupby("ID", sort=False):
        label = normalize_class_to_label(patient_df["Class"].iloc[0]) if "Class" in patient_df.columns else "N/A"
        rows.append(
            {
                "patient_id": str(patient_id),
                "class_label": label,
                "rows": int(len(patient_df)),
            }
        )
    return rows