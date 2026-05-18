from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from backend.constants import (
    CLASS_TO_LABEL,
    REQUIRED_COLUMNS,
    REQUIRED_EEG_COLUMNS,
    normalize_class_to_int as normalize_class_value,
)
from scripts.epochs import create_epochs
from scripts.features import extract_epoch_features
from scripts.preprocessing import preprocess_dataset
from scripts.signal_preprocessing import apply_basic_filtering, zscore_per_subject
from scripts.spectral_features import extract_spectral_features


__all__ = [
    "CLASS_TO_LABEL",
    "REQUIRED_COLUMNS",
    "REQUIRED_EEG_COLUMNS",
    "PreparedEpochs",
    "read_csv",
    "normalize_class_value",
    "get_dataset_stats",
    "validate_training_dataframe",
    "prepare_epochs",
    "features_for_mode",
    "n_splits_for_groups",
]


@dataclass
class PreparedEpochs:
    x_epochs: np.ndarray
    y_epochs: np.ndarray
    groups_epochs: np.ndarray
    eeg_columns: list[str]


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


def validate_training_dataframe(df: pd.DataFrame) -> None:
    missing = _missing_required_columns(df)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    if df.empty:
        raise ValueError("El dataset esta vacio.")

    for column in REQUIRED_EEG_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"La columna EEG {column} debe ser numerica.")

    normalized_labels = df["Class"].map(normalize_class_value)
    if normalized_labels.isna().any():
        raise ValueError("La columna Class contiene valores no soportados.")

    if normalized_labels.nunique() < 2:
        raise ValueError("Se necesitan pacientes de Control y TDAH para entrenar y evaluar.")


def prepare_epochs(df: pd.DataFrame, eeg_params: dict[str, Any]) -> PreparedEpochs:
    df = df.copy()
    df["Class"] = df["Class"].map(normalize_class_value)
    df, _ = preprocess_dataset(df)

    if bool(eeg_params.get("use_filtering", False)):
        sfreq = int(eeg_params.get("sfreq", 128))
        df = apply_basic_filtering(df, REQUIRED_EEG_COLUMNS, sfreq=sfreq)
        df = zscore_per_subject(df, REQUIRED_EEG_COLUMNS)

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df,
        eeg_columns=REQUIRED_EEG_COLUMNS,
        epoch_size=int(eeg_params.get("epoch_size", 1920)),
        step_size=int(eeg_params.get("step_size", 960)),
    )

    if len(x_epochs) == 0:
        raise ValueError("No se han podido crear epochs. Revisa epoch_size y step_size.")

    return PreparedEpochs(
        x_epochs=x_epochs,
        y_epochs=y_epochs.astype(int),
        groups_epochs=groups_epochs.astype(str),
        eeg_columns=REQUIRED_EEG_COLUMNS,
    )


def features_for_mode(
    x_epochs: np.ndarray,
    eeg_columns: list[str],
    eeg_params: dict[str, Any],
) -> pd.DataFrame:
    mode = eeg_params.get("feature_mode", "combined")
    sfreq = int(eeg_params.get("sfreq", 128))
    nperseg = min(int(eeg_params.get("nperseg", 960)), x_epochs.shape[1])

    if mode == "temporal":
        return extract_epoch_features(x_epochs, eeg_columns)
    if mode == "spectral":
        return extract_spectral_features(x_epochs, eeg_columns, sfreq=sfreq, nperseg=nperseg)
    if mode == "combined":
        temporal = extract_epoch_features(x_epochs, eeg_columns)
        spectral = extract_spectral_features(x_epochs, eeg_columns, sfreq=sfreq, nperseg=nperseg)
        return pd.concat([temporal, spectral], axis=1)

    raise ValueError(f"Modo de features no soportado: {mode}")


def n_splits_for_groups(y_epochs: np.ndarray, groups_epochs: np.ndarray) -> int:
    subject_labels = pd.DataFrame({"group": groups_epochs, "label": y_epochs}).groupby("group")["label"].first()
    min_subjects_per_class = int(subject_labels.value_counts().min())
    n_splits = min(5, min_subjects_per_class)

    if n_splits < 2:
        raise ValueError("Se necesitan al menos 2 pacientes por clase para aplicar CV cross-subject.")

    return n_splits


def _class_name(value: Any) -> str:
    normalized = normalize_class_value(value)
    if normalized is None:
        return str(value)
    return CLASS_TO_LABEL[normalized]


def _missing_required_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def _class_distribution(df: pd.DataFrame) -> dict[str, int]:
    if "Class" not in df.columns:
        return {}

    labels = df["Class"].map(_class_name)
    return {str(label): int(count) for label, count in labels.value_counts(dropna=False).items()}


def _patient_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "ID" not in df.columns:
        return []

    rows = []
    for patient_id, patient_df in df.groupby("ID", sort=False):
        label = _class_name(patient_df["Class"].iloc[0]) if "Class" in patient_df.columns else "N/A"
        rows.append({"patient_id": str(patient_id), "class_label": label, "rows": int(len(patient_df))})
    return rows
