from backend.model_registry import catalog
from backend.services.training_service import get_training_options


def test_catalog_display_and_family_for_known_models():
    assert catalog.display_name("random_forest") == "Random Forest"
    assert catalog.display_name("cnn_1d") == "CNN 1D"
    assert catalog.model_family("random_forest") == catalog.MACHINE_LEARNING
    assert catalog.model_family("cnn_1d") == catalog.DEEP_LEARNING
    assert catalog.model_type("xgboost") == "ml"
    assert catalog.model_type("cnn_lstm") == "dl"


def test_catalog_fallback_for_unknown_model():
    assert catalog.display_name("my_new_model") == "My New Model"
    assert catalog.model_family("desconocido", default=catalog.DEEP_LEARNING) == catalog.DEEP_LEARNING
    assert catalog.model_type("desconocido", default="dl") == "dl"


def test_training_options_display_names_come_from_catalog():
    """El display de entrenamiento debe salir del mismo catalogo que inferencia."""
    opts = get_training_options()
    models = {
        **opts["model_types"]["ml"]["models"],
        **opts["model_types"]["dl"]["models"],
    }
    for name, spec in models.items():
        assert spec["display_name"] == catalog.display_name(name)
