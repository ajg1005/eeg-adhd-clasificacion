import pandas as pd

from scripts.epochs import create_epochs
from scripts.feature_pipeline import align_feature_columns, build_features_from_config
from scripts.preprocessing import preprocess_dataset
from scripts.signal_preprocessing import apply_basic_filtering, zscore_per_subject


def map_prediction_label(prediction):
    mapping = {
        "0": "Control",
        "1": "ADHD",
        0: "Control",
        1: "ADHD",
    }

    return mapping.get(prediction, str(prediction))


def validate_eeg_dataframe(df, expected_channels):
    if df is None or df.empty:
        raise ValueError("El archivo esta vacio.")

    missing_channels = [channel for channel in expected_channels if channel not in df.columns]

    if missing_channels:
        raise ValueError(f"Faltan canales EEG esperados: {missing_channels}")

    non_numeric = []
    for column in expected_channels:
        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    if non_numeric:
        raise ValueError(f"Estas columnas EEG no son numericas: {non_numeric}")

    return True


def prepare_features_from_dataframe(df, metadata, feature_columns):
    df = df.copy()
    channels = metadata["channels"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]

    validate_eeg_dataframe(df, expected_channels=channels)

    if "Class" not in df.columns:
        df["Class"] = 0

    if "ID" not in df.columns:
        df["ID"] = "uploaded_file"

    df_clean, eeg_cols = preprocess_dataset(df)

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=epoch_size,
        step_size=step_size,
    )

    if len(x_epochs) == 0:
        raise ValueError(
            "No se han podido generar epochs. El archivo puede ser demasiado corto."
        )

    x_features = align_feature_columns(
        build_features_from_config(x_epochs, eeg_cols, metadata),
        feature_columns,
    )

    return x_features, x_epochs, y_epochs, groups_epochs


def prepare_dl_epochs_from_dataframe(df, metadata):
    df = df.copy()
    channels = metadata["channels"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]

    validate_eeg_dataframe(df, expected_channels=channels)

    if "Class" not in df.columns:
        df["Class"] = 0

    if "ID" not in df.columns:
        df["ID"] = "uploaded_file"

    df_clean, eeg_cols = preprocess_dataset(df)

    if metadata.get("apply_filtering", False):
        df_clean = apply_basic_filtering(
            df_clean,
            eeg_cols,
            subject_col="ID",
            sfreq=metadata["sfreq"],
        )

    if metadata.get("apply_zscore", False):
        df_clean = zscore_per_subject(
            df_clean,
            eeg_cols,
            subject_col="ID",
        )

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=epoch_size,
        step_size=step_size,
    )

    if len(x_epochs) == 0:
        raise ValueError(
            "No se han podido generar epochs. El archivo puede ser demasiado corto."
        )

    return x_epochs.astype("float32"), y_epochs, groups_epochs
