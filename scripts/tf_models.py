from tensorflow import keras
from tensorflow.keras import layers, regularizers


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


def build_cnn_1d(input_shape, dropout=0.5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(16, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)                          # <-- nuevo
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)                          # <-- nuevo
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4))(x)        
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_1d")


def build_cnn_lstm(input_shape, dropout=0.5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(16, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout * 0.5)(x)                          
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.5)(x)                         
    x = layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.2))(x)  
    x = layers.Dense(32, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4))(x)       
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm")


def build_model(model_name, input_shape, dropout=0.5):
    if model_name == "eegnet":
        return build_eegnet(input_shape=input_shape, dropout=dropout)
    if model_name == "cnn_1d":
        return build_cnn_1d(input_shape=input_shape, dropout=dropout)
    if model_name == "cnn_lstm":
        return build_cnn_lstm(input_shape=input_shape, dropout=dropout)

    raise ValueError(f"Modelo no soportado: {model_name}")
