import pandas as pd
from scipy.signal import butter, filtfilt


def bandpass_filter_1d(signal, sfreq, lowcut, highcut, order=4):
    nyquist = sfreq / 2.0
    highcut = min(highcut, nyquist - 1e-6)
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def apply_basic_filtering(
    df: pd.DataFrame,
    eeg_cols: list[str],
    subject_col: str = "ID",
    sfreq: int = 128,
    lowcut: float = 0.5,
    highcut: float = 50.0,
) -> pd.DataFrame:
    """
    Aplica filtrado pasabanda sobre la señal continua de cada sujeto.
    """
    df_filtered = df.copy()

    for _, group_df in df_filtered.groupby(subject_col, sort=False):
        subject_idx = group_df.index

        for col in eeg_cols:
            signal = group_df[col].to_numpy(dtype=float)
            signal = bandpass_filter_1d(
                signal,
                sfreq=sfreq,
                lowcut=lowcut,
                highcut=highcut,
            )
            df_filtered.loc[subject_idx, col] = signal

    return df_filtered


def zscore_per_subject(
    df: pd.DataFrame,
    eeg_cols: list[str],
    subject_col: str = "ID",
) -> pd.DataFrame:
    """
    Normaliza la señal continua de cada sujeto, canal a canal.
    """
    df_out = df.copy()

    for _, group_df in df.groupby(subject_col, sort=False):
        idx = group_df.index
        signal = group_df[eeg_cols].to_numpy(dtype=float)

        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        std[std == 0] = 1.0

        df_out.loc[idx, eeg_cols] = (signal - mean) / std

    return df_out