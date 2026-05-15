import pandas as pd

from scripts.epochs import create_epochs
from scripts.features import extract_epoch_features
from scripts.preprocessing import preprocess_dataset
from scripts.signal_preprocessing import apply_basic_filtering, zscore_per_subject
from scripts.spectral_features import extract_spectral_features


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
        raise ValueError("El archivo está vacío.")

    missing_channels = [ch for ch in expected_channels if ch not in df.columns]

    if missing_channels:
        raise ValueError(f"Faltan canales EEG esperados: {missing_channels}")

    non_numeric = []

    for col in expected_channels:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        raise ValueError(f"Estas columnas EEG no son numéricas: {non_numeric}")

    return True


def prepare_features_from_dataframe(df, metadata, feature_columns):
    df = df.copy()

    channels = metadata["channels"]
    sfreq = metadata["sfreq"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]
    nperseg = metadata.get("nperseg", epoch_size)
    feature_mode = metadata["feature_mode"]

    validate_eeg_dataframe(df, expected_channels=channels)

    if "Class" not in df.columns:
        df["Class"] = 0

    if "ID" not in df.columns:
        df["ID"] = "uploaded_file"

    df_clean, eeg_cols = preprocess_dataset(df)

    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=epoch_size,
        step_size=step_size,
    )

    if len(X_epochs) == 0:
        raise ValueError(
            "No se han podido generar epochs. El archivo puede ser demasiado corto."
        )

    X_time = extract_epoch_features(
        X_epochs,
        eeg_cols,
    )

    X_spectral = extract_spectral_features(
        X_epochs=X_epochs,
        channel_names=eeg_cols,
        sfreq=sfreq,
        nperseg=nperseg,
    )

    if feature_mode in ["time", "temporal"]:
        X_features = X_time
    elif feature_mode == "spectral":
        X_features = X_spectral
    elif feature_mode == "combined":
        X_features = pd.concat(
            [
                X_time.reset_index(drop=True),
                X_spectral.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"feature_mode no válido: {feature_mode}")

    missing_features = [col for col in feature_columns if col not in X_features.columns]

    if missing_features:
        raise ValueError(
            f"Faltan features esperadas por el modelo: {missing_features[:20]}"
        )

    X_features = X_features[feature_columns]

    return X_features, X_epochs, y_epochs, groups_epochs


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

    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=epoch_size,
        step_size=step_size,
    )

    if len(X_epochs) == 0:
        raise ValueError(
            "No se han podido generar epochs. El archivo puede ser demasiado corto."
        )

    return X_epochs.astype("float32"), y_epochs, groups_epochs
