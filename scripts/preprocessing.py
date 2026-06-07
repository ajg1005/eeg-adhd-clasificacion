"""Validacion y limpieza del dataset antes de segmentar en epochs."""

import pandas as pd

from scripts.constants import normalize_class_to_int


def preprocess_dataset(
    df: pd.DataFrame,
    subject_col: str = "ID",
    label_col: str = "Class",
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # Comprobar columnas obligatorias.
    required_cols = [subject_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # Eliminar filas sin identificador de sujeto o sin etiqueta.
    df = df.dropna(subset=[subject_col, label_col])

    # Unifica etiquetas textuales y numéricas antes de entrenar o evaluar.
    df[label_col] = df[label_col].map(normalize_class_to_int)
    if df[label_col].isna().any():
        unknown = df.loc[df[label_col].isna(), label_col].unique()
        raise ValueError(f"Valores de Class no reconocidos: {unknown}")
    df[label_col] = df[label_col].astype(int)

    # Variables EEG: todas menos identificador y etiqueta.
    eeg_cols = [col for col in df.columns if col not in [subject_col, label_col]]
    if not eeg_cols:
        raise ValueError("No se encontraron columnas EEG.")

    return df, eeg_cols
