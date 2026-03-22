import numpy as np
import pandas as pd


def create_epochs(
    df,
    eeg_columns,
    label_column="Class",
    group_column="ID",
    epoch_size=128,
    step_size=64,
):

    X_epochs = []
    y_epochs = []
    groups_epochs = []

    # Agrupar por sujeto para crear ventanas dentro de cada paciente
    for subject_id, group_df in df.groupby(group_column, sort=False):
     
        group_df = group_df.reset_index(drop=True)

        # Sacar la señal EEG del sujeto como array
        signal = group_df[eeg_columns].to_numpy(dtype=float)

        # Sacar las etiquetas del sujeto
        labels = group_df[label_column].to_numpy()

        # Número total de muestras del sujeto
        n_samples = len(group_df)

        # Recorrer la señal creando ventanas
        for start in range(0, n_samples - epoch_size + 1, step_size):
            end = start + epoch_size

            # Cortar la señal de esa ventana
            epoch_signal = signal[start:end]

            # Cortar también las etiquetas de esa ventana
            epoch_labels = labels[start:end]

            # Asignar una etiqueta a la epoch usando la clase más frecuente
            epoch_label = pd.Series(epoch_labels).mode().iloc[0]

            # Guardar la epoch, la etiqueta y el sujeto
            X_epochs.append(epoch_signal)
            y_epochs.append(epoch_label)
            groups_epochs.append(subject_id)

    # Devolver todo en arrays de numpy
    return np.array(X_epochs), np.array(y_epochs), np.array(groups_epochs)