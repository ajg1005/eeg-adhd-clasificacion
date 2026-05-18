import pandas as pd

from backend.modeling.model_catalog import get_model_catalog
from backend.modeling.model_factory import create_ml_model
from backend.services.training_service import get_dataset_stats, get_training_options


# comprueba que el catalogo agrupa modelos ML y DL y todos tienen parametros descritos
def test_get_model_catalog_groups_ml_and_dl_models():
    catalog = get_model_catalog()

    assert len(catalog["machine_learning"]) >= 3
    assert len(catalog["deep_learning"]) >= 2
    assert all(model["common_parameters"] for model in catalog["machine_learning"])
    assert all(model["common_parameters"] for model in catalog["deep_learning"])


# comprueba que las opciones de entrenamiento exponen los modelos UI esperados
def test_training_options_include_models_and_eeg_params():
    options = get_training_options()

    assert "ml" in options["model_types"]
    assert "dl" in options["model_types"]
    assert "xgboost" in options["model_types"]["ml"]["models"]
    assert "rbf_svc" in options["model_types"]["ml"]["models"]
    assert "random_forest" in options["model_types"]["ml"]["models"]
    assert "logistic_regression" not in options["model_types"]["ml"]["models"]
    assert "cnn_1d" in options["model_types"]["dl"]["models"]
    assert "cnn_lstm" in options["model_types"]["dl"]["models"]
    assert options["eeg_params"]["feature_mode"] == ["temporal", "spectral", "combined"]


# comprueba que la fabrica de modelos crea un pipeline de XGBoost con su paso "model"
def test_create_ml_model_xgboost_pipeline():
    model = create_ml_model("xgboost", {"n_estimators": 100, "max_depth": 4})

    assert "model" in model.named_steps


# comprueba que get_dataset_stats cuenta correctamente pacientes y clases
def test_training_dataset_stats_counts_patients_and_classes(eeg_dataframe_factory):
    rows = eeg_dataframe_factory(
        patients=[("s1", 0), ("s2", 1), ("s3", 1)],
        samples_per_patient=2,
    )
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

    stats = get_dataset_stats(csv_bytes)

    assert stats["n_patients"] == 3
    assert stats["class_distribution"] == {"ADHD": 4, "Control": 2}
