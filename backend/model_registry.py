MODEL_REGISTRY = {
    "ml_best": {
        "model_id": "ml_best",
        "display_name": "Mejor modelo ML",
        "model_family": "machine_learning",
        "description": "Modelo clasico basado en features temporales y espectrales.",
        "feature_mode": None,
        "enabled": True,
        "figures": [
            {
                "title": "Comparacion por F1",
                "url": "/figures/cv_model_comparison_f1.png",
            },
            {
                "title": "Comparacion por balanced accuracy",
                "url": "/figures/cv_model_comparison_balanced_accuracy.png",
            },
            {
                "title": "Matriz de confusion ML",
                "url": "/figures/random_forest_cv_confusion_matrix.png",
            },
            {
                "title": "Curva ROC ML",
                "url": "/figures/random_forest_cv_roc_curve.png",
            },
        ],
    },
    "dl_best": {
        "model_id": "dl_best",
        "display_name": "Mejor modelo Deep Learning",
        "model_family": "deep_learning",
        "description": "Modelo neuronal entrenado directamente sobre epochs EEG.",
        "feature_mode": "raw_epochs",
        "enabled": True,
        "figures": [
            {
                "title": "Matriz de confusion DL",
                "url": "/figures/cnn_1d_tf_cv_confusion_matrix.png",
            },
            {
                "title": "Curva ROC DL",
                "url": "/figures/cnn_1d_tf_cv_roc_curve.png",
            },
        ],
    },
}


def get_model_config(model_id: str) -> dict:
    try:
        model_config = MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError("Modelo no encontrado.") from exc

    if not model_config.get("enabled", False):
        raise ValueError("Modelo no disponible.")

    return model_config


def list_enabled_models() -> list[dict]:
    return [
        model_config
        for model_config in MODEL_REGISTRY.values()
        if model_config.get("enabled", False)
    ]
