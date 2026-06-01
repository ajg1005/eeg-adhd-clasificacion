"""Spectral EEG feature extraction based on Welch power estimates."""

import numpy as np
import pandas as pd
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def bandpower(freqs, psd, fmin, fmax):
    """
    Potencia de una banda integrando la PSD entre fmin y fmax.
    """
    idx = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(idx):
        return 0.0

    return np.trapezoid(psd[idx], freqs[idx])


def spectral_entropy(psd):
    """
    Entropia espectral a partir de la PSD normalizada.
    """
    psd = np.asarray(psd, dtype=float)

    psd_sum = np.sum(psd)

    if not np.isfinite(psd_sum) or psd_sum <= 0:
        return 0.0

    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    spectral_ent = -np.sum(psd_norm * np.log2(psd_norm))

    return spectral_ent


def mean_frequency(freqs, psd, fmin=None, fmax=None):
    """
    Calcula la frecuencia media ponderada por la PSD en un rango dado.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    if fmin is not None and fmax is not None:
        idx = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[idx]
        psd = psd[idx]

    if len(freqs) == 0 or np.sum(psd) <= 0:
        return 0.0

    mean_freq = np.sum(freqs * psd) / np.sum(psd)

    return mean_freq


def _relative_power(power, total_power):
    if total_power <= 0:
        return 0.0
    return power / total_power


def _add_band_features(row, ch_name, freqs, psd, bands, total_power):
    band_powers = {}

    for band_name, (fmin, fmax) in bands.items():
        power = bandpower(freqs, psd, fmin, fmax)
        band_powers[band_name] = power
        row[f"{ch_name}_{band_name}_abs_power"] = power
        row[f"{ch_name}_{band_name}_rel_power"] = _relative_power(power, total_power)

    return band_powers


def _add_beta_mean_frequency(row, ch_name, freqs, psd, bands):
    if ch_name not in {"O1", "O2"}:
        return

    beta_fmin, beta_fmax = bands["beta"]
    row[f"{ch_name}_beta_mean_freq"] = mean_frequency(freqs, psd, beta_fmin, beta_fmax)


def _extract_channel_spectral_features(signal, ch_name, sfreq, bands, nperseg):
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    total_idx = (freqs >= 0.5) & (freqs <= 45)
    total_psd = psd[total_idx]
    total_power = bandpower(freqs, psd, 0.5, 45)

    row = {
        f"{ch_name}_spectral_entropy": spectral_entropy(total_psd),
    }
    band_powers = _add_band_features(row, ch_name, freqs, psd, bands, total_power)

    theta = band_powers["theta"]
    beta = band_powers["beta"]
    row[f"{ch_name}_theta_beta_ratio"] = theta / beta if beta > 0 else 0.0
    _add_beta_mean_frequency(row, ch_name, freqs, psd, bands)

    return row


def extract_spectral_features(
    x_epochs,
    channel_names,
    sfreq=128,
    bands=EEG_BANDS,
    nperseg=128,
):
    """
    Extraer features espectrales por epoch y canal:
    - Potencia absoluta por banda
    - Potencia relativa por banda
    - Entropia espectral global
    - Frecuencia beta media en O1 y O2
    - Ratio theta/beta
    """

    rows = []

    for epoch in x_epochs:
        row = {}

        for ch_idx, ch_name in enumerate(channel_names):
            signal = epoch[:, ch_idx]
            row.update(
                _extract_channel_spectral_features(
                    signal,
                    ch_name,
                    sfreq,
                    bands,
                    nperseg,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)
