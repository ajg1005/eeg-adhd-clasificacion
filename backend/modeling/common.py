import pandas as pd

from scripts.epochs import create_epochs
from scripts.features import extract_epoch_features
from scripts.preprocessing import preprocess_dataset
from scripts.signal_preprocessing import apply_basic_filtering, zscore_per_subject
from scripts.spectral_features import extract_spectral_features


# Convertir la clase del modelo a un texto facil de leer
def map_prediction_label(prediction):
    mapping = {
        "0": "Control",
        "1": "ADHD",
        0: "Control",
        1: "ADHD",
    }

    return mapping.get(prediction, str(prediction))


# Validacion rapida del CSV antes de construir epochs/features
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


# Preparar un CSV subido para que tenga las mismas features que en entrenamiento ML
def prepare_features_from_dataframe(df, metadata, feature_columns):
    df = df.copy()

    channels = metadata["channels"]
    sfreq = metadata["sfreq"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]
    nperseg = metadata.get("nperseg", epoch_size)
    feature_mode = metadata["feature_mode"]

    validate_eeg_dataframe(df, expected_channels=channels)
    # En inferencia Class e ID son opcionales

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

    x_time = extract_epoch_features(
        x_epochs,
        eeg_cols,
    )

    x_spectral = extract_spectral_features(
        x_epochs=x_epochs,
        channel_names=eeg_cols,
        sfreq=sfreq,
        nperseg=nperseg,
    )

    if feature_mode in ["time", "temporal"]:
        x_features = x_time
    elif feature_mode == "spectral":
        x_features = x_spectral
    elif feature_mode == "combined":
        x_features = pd.concat(
            [
                x_time.reset_index(drop=True),
                x_spectral.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"feature_mode no válido: {feature_mode}")

    missing_features = [col for col in feature_columns if col not in x_features.columns]

    if missing_features:
        raise ValueError(
            f"Faltan features esperadas por el modelo: {missing_features[:20]}"
        )

    x_features = x_features[feature_columns]

    return x_features, x_epochs, y_epochs, groups_epochs


# Preparar epochs crudas con el mismo preprocesado usado en entrenamiento DL
def prepare_dl_epochs_from_dataframe(df, metadata):
    df = df.copy()

    channels = metadata["channels"]
    epoch_size = metadata["epoch_size"]
    step_size = metadata["step_size"]

    validate_eeg_dataframe(df, expected_channels=channels)
    # En inferencia Class e ID son opcionales

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
