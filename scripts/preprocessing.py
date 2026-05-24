import pandas as pd


def preprocess_dataset(
    df: pd.DataFrame,
    subject_col: str = "ID",
    label_col: str = "Class",
) -> tuple[pd.DataFrame, list[str]]:
    # Copia
    df = df.copy()

    # Comprobar columnas
    required_cols = [subject_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # Eliminar filas sin identificador de sujeto o sin etiqueta
    df = df.dropna(subset=[subject_col, label_col])

    # Codificar las clases si vienen como texto: "Control" -> 0, "ADHD" -> 1.
    label_map = {"Control": 0, "ADHD": 1}
    df[label_col] = df[label_col].map(label_map).fillna(df[label_col]).astype(int)

    # Variables EEG, todas menos class e id
    eeg_cols = [col for col in df.columns if col not in [subject_col, label_col]]
    if not eeg_cols:
        raise ValueError("No se encontraron columnas EEG.")

    return df, eeg_cols
