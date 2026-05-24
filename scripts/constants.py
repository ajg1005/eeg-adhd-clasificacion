"""Constantes y helpers compartidos entre research scripts y backend.

Fuente unica de verdad para:
- Semillas (RANDOM_STATE) que garantizan reproducibilidad.
- Lista de canales EEG esperados en los CSV.
- Logica de normalizacion de la columna ``Class`` (acepta variantes textuales
  y numericas).

backend/constants.py re-exporta estos simbolos por compatibilidad con los
imports existentes.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# Semilla unica usada en todos los splits, modelos y remuestreos del proyecto.
RANDOM_STATE = 42


# Canales EEG esperados en los CSV de entrada (10-20).
REQUIRED_EEG_COLUMNS: list[str] = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]
REQUIRED_COLUMNS: list[str] = REQUIRED_EEG_COLUMNS + ["Class", "ID"]


# Etiquetas legibles de las dos clases.
CLASS_ADHD = "ADHD"
CLASS_CONTROL = "Control"
CLASS_UNKNOWN = "Sin clase"

CLASS_TO_LABEL: dict[int, str] = {0: CLASS_CONTROL, 1: CLASS_ADHD}


# Aliases aceptados al normalizar la columna Class (en minusculas).
_ADHD_ALIASES = {"1", "1.0", "adhd", "tdah", "tda-h"}
_CONTROL_ALIASES = {"0", "0.0", "control", "controls", "healthy", "sano"}


def normalize_class_to_int(value: Any) -> int | None:
    """Devuelve 0 (Control), 1 (ADHD) o None si no se reconoce."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in _ADHD_ALIASES:
        return 1
    if text in _CONTROL_ALIASES:
        return 0
    return None


def normalize_class_to_label(value: Any) -> str:
    """Devuelve ADHD, Control, Sin clase o el valor original como string."""
    if pd.isna(value):
        return CLASS_UNKNOWN
    as_int = normalize_class_to_int(value)
    if as_int is not None:
        return CLASS_TO_LABEL[as_int]
    return str(value).strip()
