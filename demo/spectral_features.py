import numpy as np
import pandas as pd
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}


def bandpower_from_psd(freqs, psd, fmin, fmax):
    """
    Calcula la potencia de una banda integrando la PSD entre fmin y fmax.
    """
    idx = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(idx):
        return 0.0

    return np.trapezoid(psd[idx], freqs[idx])


def extract_spectral_features(
    X_epochs,
    channel_names,
    sfreq=128,
    bands=EEG_BANDS,
    nperseg=None,
):

    if X_epochs.ndim != 3:
        raise ValueError("X_epochs debe tener forma (n_epochs, epoch_size, n_channels)")

    n_epochs, epoch_size, n_channels = X_epochs.shape

    if len(channel_names) != n_channels:
        raise ValueError("El número de canales no coincide con channel_names")

    if nperseg is None:
        nperseg = min(epoch_size, sfreq)

    rows = []

    for epoch in X_epochs:
        row = {}

        for ch_idx, ch_name in enumerate(channel_names):
            signal = epoch[:, ch_idx]

            freqs, psd = welch(
                signal,
                fs=sfreq,
                nperseg=nperseg,
            )

            total_power = bandpower_from_psd(freqs, psd, 0.5, 45)

            for band_name, (fmin, fmax) in bands.items():
                abs_power = bandpower_from_psd(freqs, psd, fmin, fmax)
                rel_power = abs_power / total_power if total_power > 0 else 0.0

                row[f"{ch_name}_{band_name}_abs_power"] = abs_power
                row[f"{ch_name}_{band_name}_rel_power"] = rel_power

            theta_power = row[f"{ch_name}_theta_abs_power"]
            beta_power = row[f"{ch_name}_beta_abs_power"]

            row[f"{ch_name}_theta_beta_ratio"] = (
                theta_power / beta_power if beta_power > 0 else 0.0
            )

            # Opcional: frecuencia media
            if np.sum(psd) > 0:
                mean_freq = np.sum(freqs * psd) / np.sum(psd)
            else:
                mean_freq = 0.0

            row[f"{ch_name}_mean_frequency"] = mean_freq

            # Opcional: frecuencia mediana
            cumulative_power = np.cumsum(psd)
            if cumulative_power[-1] > 0:
                median_freq = freqs[np.searchsorted(cumulative_power, cumulative_power[-1] / 2)]
            else:
                median_freq = 0.0

            row[f"{ch_name}_median_frequency"] = median_freq

        rows.append(row)

    return pd.DataFrame(rows)