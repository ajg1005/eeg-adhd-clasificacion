import pandas as pd
import pytest

from backend.constants import REQUIRED_EEG_COLUMNS as EEG_CHANNELS
from backend.modeling.common import map_prediction_label, validate_eeg_dataframe


# Genera un dataframe válido usando el factory de conftest.
def _valid_eeg_dataframe(eeg_dataframe_factory, n_rows=2000):
    rows = eeg_dataframe_factory(patients=[("subject_1", 0)], samples_per_patient=n_rows)
    return pd.DataFrame(rows)


# Comprueba que un CSV válido pasa la validación.
def test_validate_eeg_dataframe_valid_csv(eeg_dataframe_factory):
    df = _valid_eeg_dataframe(eeg_dataframe_factory)

    result = validate_eeg_dataframe(df, EEG_CHANNELS)

    assert result is True


# Comprueba que se detecta un canal EEG faltante.
def test_validate_eeg_dataframe_missing_channel(eeg_dataframe_factory):
    df = _valid_eeg_dataframe(eeg_dataframe_factory).drop(columns=["Fp1"])

    with pytest.raises(ValueError, match="Faltan canales EEG"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# Comprueba que se detecta un archivo vacío.
def test_validate_eeg_dataframe_empty_file():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="archivo.*vac"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# Comprueba que se detectan canales EEG no numéricos.
def test_validate_eeg_dataframe_non_numeric_channel(eeg_dataframe_factory):
    df = _valid_eeg_dataframe(eeg_dataframe_factory)
    df["Fp1"] = ["bad_value"] * len(df)

    with pytest.raises(ValueError, match="no son num"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# Comprueba que se aceptan CSV sin columnas ID ni Class en modo inferencia.
def test_validate_eeg_dataframe_allows_optional_metadata_columns(eeg_dataframe_factory):
    df = _valid_eeg_dataframe(eeg_dataframe_factory).drop(columns=["ID", "Class"])

    result = validate_eeg_dataframe(df, EEG_CHANNELS)

    assert result is True


# Comprueba el mapeo de la clase Control.
def test_map_prediction_label_control():
    assert map_prediction_label(0) == "Control"
    assert map_prediction_label("0") == "Control"


# Comprueba el mapeo de la clase ADHD.
def test_map_prediction_label_adhd():
    assert map_prediction_label(1) == "ADHD"
    assert map_prediction_label("1") == "ADHD"


# Comprueba que una clase desconocida no rompe el mapeo.
def test_map_prediction_label_unknown_value():
    assert map_prediction_label(2) == "2"
