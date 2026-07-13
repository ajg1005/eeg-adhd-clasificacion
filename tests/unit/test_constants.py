"""Tests de normalizacion de clase: cubre los aliases que acepta el sistema."""
import numpy as np
import pytest

from scripts.constants import (
    CLASS_ADHD,
    CLASS_CONTROL,
    CLASS_UNKNOWN,
    normalize_class_to_int,
    normalize_class_to_label,
)


# comprueba que distintos formatos del valor ADHD se mapean al entero 1
@pytest.mark.parametrize(
    "value",
    [1, "1", "1.0", "ADHD", "adhd", "TDAH", "tdah", "Tda-H"],
)
def test_normalize_class_to_int_recognises_adhd_aliases(value):
    assert normalize_class_to_int(value) == 1


# comprueba que distintos formatos del valor Control se mapean al entero 0
@pytest.mark.parametrize(
    "value",
    [0, "0", "0.0", "Control", "control", "CONTROLS", "Healthy", "sano"],
)
def test_normalize_class_to_int_recognises_control_aliases(value):
    assert normalize_class_to_int(value) == 0


# comprueba que un valor desconocido devuelve None (sin lanzar excepcion)
@pytest.mark.parametrize("value", ["unknown", "x", "2", "ADHD-positive"])
def test_normalize_class_to_int_returns_none_for_unknown(value):
    assert normalize_class_to_int(value) is None


# comprueba que los valores nulos (NaN, None) se tratan como desconocidos
def test_normalize_class_to_int_handles_nan():
    assert normalize_class_to_int(np.nan) is None
    assert normalize_class_to_int(None) is None


# comprueba la conversion a etiqueta legible (ADHD / Control / Sin clase)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, CLASS_ADHD),
        ("ADHD", CLASS_ADHD),
        ("tdah", CLASS_ADHD),
        (0, CLASS_CONTROL),
        ("control", CLASS_CONTROL),
        (np.nan, CLASS_UNKNOWN),
    ],
)
def test_normalize_class_to_label_maps_correctly(value, expected):
    assert normalize_class_to_label(value) == expected


# comprueba que un valor desconocido se devuelve tal cual como string
def test_normalize_class_to_label_falls_back_to_original_string():
    assert normalize_class_to_label("custom_label") == "custom_label"
