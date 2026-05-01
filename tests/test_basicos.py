from pathlib import Path
import sys

import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from inference import validate_eeg_dataframe, map_prediction_label


EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"
]


# dataframe valido
def create_valid_eeg_dataframe(n_rows=2000):
    data = {
        channel: [0.1] * n_rows
        for channel in EEG_CHANNELS
    }

    data["ID"] = ["subject_1"] * n_rows
    data["Class"] = [0] * n_rows

    return pd.DataFrame(data)


# comprueba que un csv valido pasa la validacion
def test_validate_eeg_dataframe_valid_csv():
    df = create_valid_eeg_dataframe()

    result = validate_eeg_dataframe(df, EEG_CHANNELS)

    assert result is True


# comprueba que se detecta un canal eeg faltante
def test_validate_eeg_dataframe_missing_channel():
    df = create_valid_eeg_dataframe()
    df = df.drop(columns=["Fp1"])

    with pytest.raises(ValueError, match="Faltan canales EEG"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# comprueba que se detecta un archivo vacio
def test_validate_eeg_dataframe_empty_file():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="El archivo está vacío"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# comprueba que se detectan canales no numericos
def test_validate_eeg_dataframe_non_numeric_channel():
    df = create_valid_eeg_dataframe()
    df["Fp1"] = ["bad_value"] * len(df)

    with pytest.raises(ValueError, match="no son numéricas"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# comprueba el mapeo de la clase control
def test_map_prediction_label_control():
    assert map_prediction_label(0) == "Control"
    assert map_prediction_label("0") == "Control"


# comprueba el mapeo de la clase adhd
def test_map_prediction_label_adhd():
    assert map_prediction_label(1) == "ADHD"
    assert map_prediction_label("1") == "ADHD"