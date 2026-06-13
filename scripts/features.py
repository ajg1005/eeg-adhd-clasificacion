"""Caracteristicas temporales por epoch y canal EEG."""

import numpy as np
import pandas as pd


def extract_epoch_features(x_epochs, channel_names):
    """Extrae caracteristicas estadisticas temporales por canal y epoch.

    Por cada epoch saca 12 medidas por canal (mean, median, std, var, min,
    max, range, q25, q75, iqr, energy, rms). Con 19 canales 10-20 salen 228
    features por epoch. Son la entrada de los modelos ML clasicos.
    """

    # Comprobar dimensiones
    if x_epochs.ndim != 3:
        raise ValueError("X_epochs debe tener forma (n_epochs, epoch_size, n_channels)")

    _, _, n_channels = x_epochs.shape

    if len(channel_names) != n_channels:
        raise ValueError("El número de canales no coincide con channel_names")

    # Diccionario por ventana 
    rows = []

    # Iterar por epoch y guardar las caracteristicas
    for epoch in x_epochs:
        
        row = {}

        # Iterar por canal
        for i, ch in enumerate(channel_names):
            
            signal = epoch[:, i]

            # Caracteristicas basicas
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            min_val = np.min(signal)
            max_val = np.max(signal)
            median_val = np.median(signal)
            var_val = np.var(signal)

            # Percentiles
            q25_val = np.percentile(signal, 25)
            q75_val = np.percentile(signal, 75)
            iqr_val = q75_val - q25_val

            # Rango
            range_val = max_val - min_val

            energy_val = np.sum(signal ** 2)
            rms_val = np.sqrt(np.mean(signal ** 2))

            # Guardar caracteristicas en diccionario con nombres
            row[f"{ch}_mean"] = mean_val
            row[f"{ch}_median"] = median_val
            row[f"{ch}_std"] = std_val
            row[f"{ch}_var"] = var_val
            row[f"{ch}_min"] = min_val
            row[f"{ch}_max"] = max_val
            row[f"{ch}_range"] = range_val
            row[f"{ch}_q25"] = q25_val
            row[f"{ch}_q75"] = q75_val
            row[f"{ch}_iqr"] = iqr_val
            row[f"{ch}_energy"] = energy_val
            row[f"{ch}_rms"] = rms_val

        # Añadir
        rows.append(row)

    # Convertir en dataframe
    return pd.DataFrame(rows)

#LSTM-->Trabajar con la epoch entera
