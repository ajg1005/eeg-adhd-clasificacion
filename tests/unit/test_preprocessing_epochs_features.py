import numpy as np
import pandas as pd
import pytest
from scripts.constants import REQUIRED_EEG_COLUMNS as EEG_CHANNELS
from scripts.epochs import create_epochs
from scripts.features import extract_epoch_features
from scripts.preprocessing import preprocess_dataset


# comprueba que el preprocesado no toca etiquetas que ya son numericas
def test_preprocess_dataset_keeps_numeric_labels(eeg_dataframe_factory):
    df = pd.DataFrame(eeg_dataframe_factory(patients=[("s1", 0)], samples_per_patient=3))
    df["Class"] = [0, 1, 0]

    df_clean, eeg_cols = preprocess_dataset(df)

    assert df_clean["Class"].tolist() == [0, 1, 0]
    assert eeg_cols == EEG_CHANNELS


# comprueba que el preprocesado codifica "Control" y "ADHD" como 0 y 1
def test_preprocess_dataset_encodes_text_labels(eeg_dataframe_factory):
    df = pd.DataFrame(eeg_dataframe_factory(patients=[("s1", 0)], samples_per_patient=2))
    df["Class"] = ["Control", "ADHD"]

    df_clean, _ = preprocess_dataset(df)

    assert df_clean["Class"].tolist() == [0, 1]


# comprueba que create_epochs respeta sujeto, tamano de ventana y solapamiento
def test_create_epochs_groups_by_subject_and_step_size():
    df = pd.DataFrame(
        {
            "ID": ["s1"] * 5 + ["s2"] * 5,
            "Class": [0] * 5 + [1] * 5,
            "Fp1": list(range(5)) + list(range(10, 15)),
            "Fp2": list(range(5, 10)) + list(range(15, 20)),
        }
    )

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df,
        eeg_columns=["Fp1", "Fp2"],
        epoch_size=3,
        step_size=2,
    )

    assert x_epochs.shape == (4, 3, 2)
    assert y_epochs.tolist() == [0, 0, 1, 1]
    assert groups_epochs.tolist() == ["s1", "s1", "s2", "s2"]
    assert x_epochs[1, :, 0].tolist() == [2, 3, 4]


# comprueba algunas features temporales basicas (media, min, max, rango)
def test_extract_epoch_features_basic_values():
    x_epochs = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    features = extract_epoch_features(x_epochs, ["Fp1", "Fp2"])

    assert features.loc[0, "Fp1_mean"] == pytest.approx(3.0)
    assert features.loc[0, "Fp1_min"] == pytest.approx(1.0)
    assert features.loc[0, "Fp1_max"] == pytest.approx(5.0)
    assert features.loc[0, "Fp2_range"] == pytest.approx(4.0)
