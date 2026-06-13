"""Arquitecturas TF/Keras evaluadas en los experimentos DL."""

import keras
from keras import layers, regularizers


def build_eegnet(input_shape, dropout=0.5):
    """
    Variante ligera tipo EEGNet adaptada a entrada 3D:
    (n_times, n_channels)
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv1D(
        8,
        kernel_size=32,
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv1D(
        16,
        kernel_size=16,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling1D(pool_size=4)(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.SeparableConv1D(
        32,
        kernel_size=8,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling1D(pool_size=4)(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="eegnet")


def build_cnn_1d(input_shape, filters=16, dropout=0.5, dense_units=32):
    """CNN-1D progresiva pensada para clasificar epochs EEG cortos.

    Tres bloques convolucionales con filtros crecientes (16, 32, 64) seguidos
    de pooling, batch norm y dropout. Termina con global average pooling +
    dense + sigmoid. Diseñada para capturar patrones locales en la señal sin
    explotar en parametros.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    x = layers.Conv1D(filters * 2, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    x = layers.Conv1D(filters * 4, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_1d")


def build_cnn_lstm(input_shape, filters=16, dropout=0.5, lstm_units=32, dense_units=32):
    """Hibrido CNN + LSTM bidireccional para capturar tambien la temporalidad.

    Dos bloques Conv1D para sacar features locales y un BiLSTM encima para
    modelar la dependencia temporal en ambas direcciones. La idea es ver si
    la informacion secuencial mejora respecto a la CNN-1D pura.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    x = layers.Conv1D(filters * 2, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.5)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, recurrent_dropout=0.2))(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm")


def build_model(model_name, input_shape, **kwargs):
    """Factoria de redes DL: devuelve la arquitectura segun el nombre.

    Centraliza el dispatch para que el resto del codigo no tenga que conocer
    cada `build_*`. Si se añade una arquitectura nueva, solo hay que
    registrarla aqui.
    """
    if model_name == "eegnet":
        return build_eegnet(input_shape=input_shape, **kwargs)
    if model_name == "cnn_1d":
        return build_cnn_1d(input_shape=input_shape, **kwargs)
    if model_name == "cnn_lstm":
        return build_cnn_lstm(input_shape=input_shape, **kwargs)

    raise ValueError(f"Modelo no soportado: {model_name}")
