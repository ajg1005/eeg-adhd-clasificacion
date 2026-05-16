from typing import Any


DL_MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "cnn_1d": {
        "display_name": "CNN 1D",
        "default_params": {"filters": 32, "dropout": 0.3},
        "parameters": {
            "filters": [16, 32, 64],
            "dropout": [0.2, 0.3, 0.5],
        },
    },
    "cnn_lstm": {
        "display_name": "CNN-LSTM",
        "default_params": {"filters": 32, "dropout": 0.3, "lstm_units": 64},
        "parameters": {
            "filters": [16, 32, 64],
            "dropout": [0.2, 0.3, 0.5],
            "lstm_units": [32, 64, 128],
        },
    },
}


def _keras_modules():
    import keras
    from keras import layers

    return keras, layers


def _model_params(model_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    options = DL_MODEL_OPTIONS.get(model_name, {})
    defaults = options.get("default_params", {})
    return {**defaults, **(params or {})}


# Crear redes ligeras para entrenamiento interactivo sin tocar los modelos exportados.
def create_dl_model(
    model_name: str,
    input_shape: tuple[int, int],
    model_params: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
):
    keras, layers = _keras_modules()
    model_params = _model_params(model_name, model_params)
    training_params = training_params or {}

    filters = int(model_params.get("filters", 32))
    kernel_size = int(model_params.get("kernel_size", 5))
    dropout = float(model_params.get("dropout", 0.3))
    dense_units = int(model_params.get("dense_units", 64))
    learning_rate = float(training_params.get("learning_rate", 0.001))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    if model_name == "cnn_1d":
        x = layers.Conv1D(filters * 2, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif model_name == "cnn_lstm":
        lstm_units = int(model_params.get("lstm_units", 64))
        x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    else:
        raise ValueError(f"Modelo DL no soportado: {model_name}")

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def create_early_stopping(patience: int = 5):
    keras, _ = _keras_modules()
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True,
    )


