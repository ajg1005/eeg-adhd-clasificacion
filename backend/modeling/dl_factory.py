from typing import Any


DL_MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "cnn_1d": {
        "display_name": "CNN 1D",
        "default_params": {"filters": 16, "dropout": 0.4},
        "parameters": {
            "filters": [16, 32, 64],
            "dropout": [0.2, 0.3, 0.4, 0.5],
        },
    },
    "cnn_lstm": {
        "display_name": "CNN-LSTM",
        "default_params": {"filters": 16, "dropout": 0.4, "lstm_units": 32},
        "parameters": {
            "filters": [16, 32, 64],
            "dropout": [0.2, 0.3, 0.4, 0.5],
            "lstm_units": [32, 64, 128],
        },
    },
}


def _keras_module():
    import keras

    return keras


def _model_params(model_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    options = DL_MODEL_OPTIONS.get(model_name, {})
    defaults = options.get("default_params", {})
    return {**defaults, **(params or {})}


# Crear redes ligeras para entrenamiento interactivo reutilizando las arquitecturas
# de scripts/tf_models y compilando con los parametros de entrenamiento de la UI.
def create_dl_model(
    model_name: str,
    input_shape: tuple[int, int],
    model_params: dict[str, Any] | None = None,
    training_params: dict[str, Any] | None = None,
):
    from scripts.tf_models import build_model

    keras = _keras_module()
    model_params = _model_params(model_name, model_params)
    training_params = training_params or {}

    model = build_model(model_name, input_shape, **model_params)

    learning_rate = float(training_params.get("learning_rate", 0.0003))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def create_early_stopping(patience: int = 4):
    keras = _keras_module()
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True,
    )
