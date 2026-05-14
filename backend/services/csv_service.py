import pandas as pd
from fastapi import UploadFile


def ensure_csv_upload(file: UploadFile) -> None:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Solo se admiten archivos CSV.")


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    ensure_csv_upload(file)
    file.file.seek(0)
    return pd.read_csv(file.file)


def get_available_channels(df: pd.DataFrame, expected_channels: list[str]) -> list[str]:
    return [channel for channel in expected_channels if channel in df.columns]


def build_signal_preview(
    df: pd.DataFrame,
    expected_channels: list[str],
    channel: str,
    max_points: int,
) -> dict:
    if channel not in expected_channels:
        raise ValueError(f"El canal {channel} no esta entre los canales esperados.")

    if channel not in df.columns:
        raise ValueError(f"El archivo no contiene el canal {channel}.")

    n_points = min(max_points, len(df))
    signal = df[channel].iloc[:n_points].astype(float).tolist()

    return {
        "channel": channel,
        "n_points": int(n_points),
        "samples": [
            {
                "sample": index,
                "value": value,
            }
            for index, value in enumerate(signal)
        ],
    }
