import numpy as np
import pandas as pd
import pytest

from backend.constants import REQUIRED_EEG_COLUMNS as EEG_CHANNELS
from backend.modeling.common import map_prediction_label, validate_eeg_dataframe
from backend.services.dataset_service import build_dataset_summary
from backend.modeling.model_catalog import get_model_catalog
from backend.modeling.model_factory import create_ml_model
from backend.services.training_service import get_dataset_stats, get_training_options
from scripts.epochs import create_epochs
from scripts.features import extract_epoch_features
from scripts.preprocessing import preprocess_dataset


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

    with pytest.raises(ValueError, match="archivo.*vac"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# comprueba que se detectan canales no numericos
def test_validate_eeg_dataframe_non_numeric_channel():
    df = create_valid_eeg_dataframe()
    df["Fp1"] = ["bad_value"] * len(df)

    with pytest.raises(ValueError, match="no son num"):
        validate_eeg_dataframe(df, EEG_CHANNELS)


# comprueba el mapeo de la clase control
def test_map_prediction_label_control():
    assert map_prediction_label(0) == "Control"
    assert map_prediction_label("0") == "Control"


# comprueba el mapeo de la clase adhd
def test_map_prediction_label_adhd():
    assert map_prediction_label(1) == "ADHD"
    assert map_prediction_label("1") == "ADHD"


# comprueba que una clase desconocida no rompe el mapeo
def test_map_prediction_label_unknown_value():
    assert map_prediction_label(2) == "2"


# comprueba que se aceptan archivos sin ID ni Class
def test_validate_eeg_dataframe_allows_optional_metadata_columns():
    df = create_valid_eeg_dataframe().drop(columns=["ID", "Class"])

    result = validate_eeg_dataframe(df, EEG_CHANNELS)

    assert result is True


# comprueba que el preprocesado mantiene clases numericas
def test_preprocess_dataset_keeps_numeric_labels():
    df = create_valid_eeg_dataframe(n_rows=3)
    df["Class"] = [0, 1, 0]

    df_clean, eeg_cols = preprocess_dataset(df)

    assert df_clean["Class"].tolist() == [0, 1, 0]
    assert eeg_cols == EEG_CHANNELS


# comprueba que el preprocesado codifica clases en texto
def test_preprocess_dataset_encodes_text_labels():
    df = create_valid_eeg_dataframe(n_rows=2)
    df["Class"] = ["Control", "ADHD"]

    df_clean, _ = preprocess_dataset(df)

    assert df_clean["Class"].tolist() == [0, 1]


# comprueba que las epochs respetan sujeto, tamano y salto
def test_create_epochs_groups_by_subject_and_step_size():
    df = pd.DataFrame(
        {
            "ID": ["s1"] * 5 + ["s2"] * 5,
            "Class": [0] * 5 + [1] * 5,
            "Fp1": list(range(5)) + list(range(10, 15)),
            "Fp2": list(range(5, 10)) + list(range(15, 20)),
        }
    )

    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df,
        eeg_columns=["Fp1", "Fp2"],
        epoch_size=3,
        step_size=2,
    )

    assert X_epochs.shape == (4, 3, 2)
    assert y_epochs.tolist() == [0, 0, 1, 1]
    assert groups_epochs.tolist() == ["s1", "s1", "s2", "s2"]
    assert X_epochs[1, :, 0].tolist() == [2, 3, 4]


# comprueba algunas features temporales basicas
def test_extract_epoch_features_basic_values():
    X_epochs = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    features = extract_epoch_features(X_epochs, ["Fp1", "Fp2"])

    assert features.loc[0, "Fp1_mean"] == 3.0
    assert features.loc[0, "Fp1_min"] == 1.0
    assert features.loc[0, "Fp1_max"] == 5.0
    assert features.loc[0, "Fp2_range"] == 4.0


# comprueba estadisticas basicas del dataset cargado
def test_build_dataset_summary_counts_classes_and_patients():
    df = pd.DataFrame(
        {
            "ID": ["s1", "s1", "s2", "s2", "s3"],
            "Class": [0, 0, 1, 1, "ADHD"],
            "Fp1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Fp2": [1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )

    summary = build_dataset_summary(df, class_filter="all", max_patients=2)

    assert summary["rows"] == 5
    assert summary["n_eeg_channels"] == 2
    assert summary["total_patients"] == 3
    assert summary["shown_patients_count"] == 2
    assert summary["class_counts"] == {"ADHD": 3, "Control": 2}


# comprueba que se pueden mostrar solo pacientes tdah
def test_build_dataset_summary_filters_adhd_patients():
    df = pd.DataFrame(
        {
            "ID": ["s1", "s1", "s2", "s2", "s3"],
            "Class": [0, 0, 1, 1, "ADHD"],
            "Fp1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    summary = build_dataset_summary(df, class_filter="adhd", max_patients=10)

    assert summary["filtered_patients_count"] == 2
    assert [patient["patient_id"] for patient in summary["patients"]] == ["s2", "s3"]
    assert all(patient["class_label"] == "ADHD" for patient in summary["patients"])

# comprueba que el catalogo separa modelos ml y dl con parametros
def test_get_model_catalog_groups_ml_and_dl_models():
    catalog = get_model_catalog()

    assert len(catalog["machine_learning"]) >= 3
    assert len(catalog["deep_learning"]) >= 2
    assert all(model["common_parameters"] for model in catalog["machine_learning"])
    assert all(model["common_parameters"] for model in catalog["deep_learning"])

# comprueba que las opciones de entrenamiento interactivo exponen ML, DL y parametros EEG
def test_training_options_include_models_and_eeg_params():
    options = get_training_options()

    assert "ml" in options["model_types"]
    assert "dl" in options["model_types"]
    assert "xgboost" in options["model_types"]["ml"]["models"]
    assert "rbf_svc" in options["model_types"]["ml"]["models"]
    assert "random_forest" in options["model_types"]["ml"]["models"]
    assert "logistic_regression" not in options["model_types"]["ml"]["models"]
    assert "cnn_1d" in options["model_types"]["dl"]["models"]
    assert "cnn_lstm" in options["model_types"]["dl"]["models"]
    assert options["eeg_params"]["feature_mode"] == ["temporal", "spectral", "combined"]


# comprueba que la fabrica ML crea el pipeline de XGBoost
def test_create_ml_model_xgboost_pipeline():
    model = create_ml_model("xgboost", {"n_estimators": 100, "max_depth": 4})

    assert "model" in model.named_steps







# comprueba estadisticas del dataset de entrenamiento
def test_training_dataset_stats_counts_patients_and_classes():
    df = create_valid_eeg_dataframe(n_rows=6)
    df["ID"] = ["s1", "s1", "s2", "s2", "s3", "s3"]
    df["Class"] = [0, 0, 1, 1, 1, 1]
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    stats = get_dataset_stats(csv_bytes)

    assert stats["n_patients"] == 3
    assert stats["class_distribution"] == {"ADHD": 4, "Control": 2}
