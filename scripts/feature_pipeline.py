"""Pipeline comun para construir y alinear features EEG."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

try:
    from scripts.features import extract_epoch_features
    from scripts.spectral_features import extract_spectral_features
except ModuleNotFoundError:  # Permite ejecutar scripts/*.py directamente.
    from features import extract_epoch_features
    from spectral_features import extract_spectral_features


FEATURE_MODE_ALIASES = {
    "time": "temporal",
    "temporal": "temporal",
    "spectral": "spectral",
    "combined": "combined",
}


def normalize_feature_mode(feature_mode: str | None) -> str:
    mode = FEATURE_MODE_ALIASES.get(str(feature_mode or "combined").lower())
    if mode is None:
        raise ValueError(f"feature_mode no soportado: {feature_mode}")
    return mode


def build_features_from_epochs(
    x_epochs,
    channel_names: list[str],
    feature_mode: str = "combined",
    sfreq: int | float = 128,
    nperseg: int | None = None,
) -> pd.DataFrame:
    mode = normalize_feature_mode(feature_mode)
    segment_size = _safe_nperseg(x_epochs, nperseg)

    if mode == "temporal":
        return extract_epoch_features(x_epochs, channel_names)

    if mode == "spectral":
        return extract_spectral_features(
            x_epochs=x_epochs,
            channel_names=channel_names,
            sfreq=sfreq,
            nperseg=segment_size,
        )

    temporal = extract_epoch_features(x_epochs, channel_names)
    spectral = extract_spectral_features(
        x_epochs=x_epochs,
        channel_names=channel_names,
        sfreq=sfreq,
        nperseg=segment_size,
    )

    return pd.concat(
        [temporal.reset_index(drop=True), spectral.reset_index(drop=True)],
        axis=1,
    )


def build_features_from_config(x_epochs, channel_names: list[str], config: dict[str, Any]) -> pd.DataFrame:
    return build_features_from_epochs(
        x_epochs=x_epochs,
        channel_names=channel_names,
        feature_mode=config.get("feature_mode", "combined"),
        sfreq=config.get("sfreq", 128),
        nperseg=config.get("nperseg", config.get("epoch_size")),
    )


def align_feature_columns(x_features: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    expected_columns = list(feature_columns)
    missing_features = [column for column in expected_columns if column not in x_features.columns]

    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(f"Faltan features esperadas por el modelo: {preview}")

    return x_features.loc[:, expected_columns]


def _safe_nperseg(x_epochs, nperseg: int | None) -> int:
    epoch_size = int(x_epochs.shape[1])
    if nperseg is None:
        return epoch_size
    return min(int(nperseg), epoch_size)
