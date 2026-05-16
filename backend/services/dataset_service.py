import pandas as pd

from backend.constants import normalize_class_to_label


# Resumir filas, columnas, clases, canales y pacientes del CSV cargado
def build_dataset_summary(
    df: pd.DataFrame,
    class_filter: str = "all",
    max_patients: int = 10,
) -> dict:
    if df is None or df.empty:
        raise ValueError("El archivo esta vacio.")

    max_patients = max(1, min(int(max_patients), 100))
    class_filter = (class_filter or "all").lower()

    if "Class" in df.columns:
        normalized_classes = df["Class"].map(normalize_class_to_label)
        class_counts = {
            label: int(count)
            for label, count in normalized_classes.value_counts(dropna=False).items()
        }
    else:
        class_counts = {}

    eeg_channels = [
        column
        for column in df.columns
        if column not in {"ID", "Class"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    patients = []
    total_patients = 0
    filtered_patients_count = 0

    if "ID" in df.columns:
        patient_rows = []
        for patient_id, group_df in df.groupby("ID", sort=False):
            patient_class = "Sin clase"
            if "Class" in group_df.columns:
                patient_class = normalize_class_to_label(group_df["Class"].iloc[0])

            patient_rows.append(
                {
                    "patient_id": str(patient_id),
                    "class_label": patient_class,
                    "n_samples": int(len(group_df)),
                }
            )

        total_patients = len(patient_rows)

        if class_filter != "all":
            patient_rows = [
                patient
                for patient in patient_rows
                if patient["class_label"].lower() == class_filter
            ]

        filtered_patients_count = len(patient_rows)
        patients = patient_rows[:max_patients]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "has_id": "ID" in df.columns,
        "has_class": "Class" in df.columns,
        "class_counts": class_counts,
        "eeg_channels": eeg_channels,
        "n_eeg_channels": int(len(eeg_channels)),
        "total_patients": int(total_patients),
        "filtered_patients_count": int(filtered_patients_count),
        "shown_patients_count": int(len(patients)),
        "patients": patients,
        "class_filter": class_filter,
        "max_patients": int(max_patients),
    }
