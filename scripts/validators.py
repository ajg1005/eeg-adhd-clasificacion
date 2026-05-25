"""Validacion comun de DataFrames EEG.

Centraliza las comprobaciones basicas (estructura, tipos, canales) que antes
estaban repartidas entre backend/modeling/common.py y
backend/services/training_data.py.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from scripts.constants import REQUIRED_EEG_COLUMNS, normalize_class_to_int
except ModuleNotFoundError:
    from constants import REQUIRED_EEG_COLUMNS, normalize_class_to_int


# Comprueba que el DataFrame contiene los canales EEG esperados y que son numéricos.
def validate_eeg_dataframe(df: pd.DataFrame, expected_channels: Iterable[str]) -> bool:
    if df is None or df.empty:
        raise ValueError("El archivo está vacío.")

    expected = list(expected_channels)
    missing_channels = [channel for channel in expected if channel not in df.columns]
    if missing_channels:
        raise ValueError(f"Faltan canales EEG esperados: {missing_channels}")

    non_numeric = [
        column for column in expected
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Estas columnas EEG no son numéricas: {non_numeric}")

    return True


# Validación estricta para datasets de entrenamiento.
def validate_training_dataframe(df: pd.DataFrame) -> None:
    validate_eeg_dataframe(df, REQUIRED_EEG_COLUMNS)

    missing = [column for column in ("Class", "ID") if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    normalized_labels = df["Class"].map(normalize_class_to_int)
    if normalized_labels.isna().any():
        raise ValueError("La columna Class contiene valores no soportados.")

    if normalized_labels.nunique() < 2:
        raise ValueError("Se necesitan pacientes de Control y TDAH para entrenar y evaluar.")
