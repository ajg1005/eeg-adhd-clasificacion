import pandas as pd
from fastapi import UploadFile


# Comprobar que el archivo subido es un CSV
def ensure_csv_upload(file: UploadFile) -> None:
    """Raise ValueError when the FastAPI upload is not a CSV file."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Solo se admiten archivos CSV.")


# Leer el CSV subido desde FastAPI
def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Validate and parse an uploaded CSV into a pandas DataFrame."""
    ensure_csv_upload(file)
    file.file.seek(0)
    return pd.read_csv(file.file)
