"""Tests de consistencia entre scripts/pipeline (fuente de verdad) y
backend/modeling/model_factory (wrapper para la UI).

Estos tests garantizan que el refactor de unificacion sigue vigente:
- scripts.pipeline.create_ml_model soporta los 5 modelos (research).
- backend.modeling.model_factory.create_ml_model solo expone los 3 de UI.
- Para los modelos UI, ambas factories producen pipelines equivalentes.
"""
import pytest

from backend.modeling.model_factory import ML_MODEL_OPTIONS
from backend.modeling.model_factory import create_ml_model as backend_create
from scripts.pipeline import (
    ALL_MODEL_NAMES,
    create_ml_model as scripts_create,
    get_models,
)


UI_MODELS = ["rbf_svc", "random_forest", "xgboost"]
RESEARCH_ONLY_MODELS = ["logistic_regression", "knn"]


# comprueba que scripts/pipeline construye los 5 modelos sin error
@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
def test_scripts_pipeline_supports_all_models(model_name):
    pipeline = scripts_create(model_name)
    assert "model" in pipeline.named_steps


# comprueba que el backend acepta los 3 modelos expuestos en la UI
@pytest.mark.parametrize("model_name", UI_MODELS)
def test_backend_factory_supports_ui_models(model_name):
    pipeline = backend_create(model_name)
    assert "model" in pipeline.named_steps


# comprueba que el backend rechaza los modelos que solo son de research
@pytest.mark.parametrize("model_name", RESEARCH_ONLY_MODELS)
def test_backend_factory_rejects_non_ui_models(model_name):
    with pytest.raises(ValueError, match="no disponible en la UI"):
        backend_create(model_name)


# comprueba que get_models devuelve los 5 modelos como pipelines listos
def test_get_models_returns_all_five():
    models = get_models()
    assert set(models.keys()) == set(ALL_MODEL_NAMES)
    for pipeline in models.values():
        assert "model" in pipeline.named_steps


# comprueba que backend (defaults UI) y scripts (mismos params) generan el mismo modelo
@pytest.mark.parametrize("model_name", UI_MODELS)
def test_factory_consistency_with_ui_defaults(model_name):
    ui_defaults = ML_MODEL_OPTIONS[model_name]["default_params"]
    backend_model = backend_create(model_name)
    scripts_model = scripts_create(model_name, ui_defaults)

    backend_estimator = backend_model.named_steps["model"]
    scripts_estimator = scripts_model.named_steps["model"]

    for param_name in ui_defaults:
        backend_value = getattr(backend_estimator, param_name)
        scripts_value = getattr(scripts_estimator, param_name)
        assert backend_value == scripts_value, (
            f"Discrepancia en {model_name}.{param_name}: "
            f"backend={backend_value!r} vs scripts={scripts_value!r}"
        )


# comprueba que el string "none" del frontend se traduce a None de Python
def test_backend_clean_params_converts_none_string():
    pipeline = backend_create("random_forest", {"class_weight": "none"})
    rf = pipeline.named_steps["model"]
    assert rf.class_weight is None
