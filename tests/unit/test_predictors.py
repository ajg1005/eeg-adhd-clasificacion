from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.inference import predictors
from backend.inference.predictors import (
    DLPredictor,
    MLPredictor,
    get_model_config,
    get_predictor,
)


class FakeMLModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([0, 1, 1])

    def predict_proba(self, X):
        return np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )


class FakeDLModel:
    def predict(self, X, batch_size=32, verbose=0):
        return np.array([[0.2], [0.8], [0.9]])


def _ml_predictor(monkeypatch):
    model_config = {
        "model_id": "ml_best",
        "display_name": "ML",
        "model_family": "machine_learning",
        "feature_mode": None,
    }
    metadata = {
        "model_name": "fake_rf",
        "feature_mode": "temporal",
        "sfreq": 128,
        "epoch_size": 2,
        "step_size": 2,
        "channels": ["Fp1"],
    }
    predictor = MLPredictor(model_config)
    monkeypatch.setattr(
        predictor,
        "load_artifacts",
        lambda: (FakeMLModel(), ["f1"], metadata, {"cv_metrics": {"f1": 1.0}}),
    )
    return predictor


def _dl_predictor(monkeypatch):
    model_config = {
        "model_id": "dl_best",
        "display_name": "DL",
        "model_family": "deep_learning",
        "feature_mode": "raw_epochs",
    }
    metadata = {
        "model_name": "fake_cnn",
        "sfreq": 128,
        "epoch_size": 2,
        "step_size": 2,
        "channels": ["Fp1"],
        "threshold": 0.5,
    }
    predictor = DLPredictor(model_config)
    monkeypatch.setattr(
        predictor,
        "load_artifacts",
        lambda: (FakeDLModel(), metadata, {"cv_metrics": {"f1": 1.0}}),
    )
    return predictor


def _trained_model_record(tmp_path, *, model_family="machine_learning"):
    artifact_path = tmp_path / "model.joblib"
    feature_columns_path = tmp_path / "feature_columns.json"
    artifact_path.write_bytes(b"fake-model")
    feature_columns_path.write_text("[]", encoding="utf-8")
    return SimpleNamespace(
        id=7,
        experiment_id=3,
        model_name="random_forest",
        model_family=model_family,
        artifact_path=str(artifact_path),
        feature_columns_path=str(feature_columns_path),
        threshold=None,
        model_metadata={"model_name": "random_forest", "feature_mode": "combined"},
    )


def test_get_model_config_rejects_unknown_model():
    with pytest.raises(ValueError, match="Modelo no encontrado"):
        get_model_config("missing")


def test_get_model_config_resolves_registered_trained_model(monkeypatch, tmp_path):
    record = _trained_model_record(tmp_path)
    monkeypatch.setattr(
        predictors,
        "get_trained_model",
        lambda trained_model_id: record if trained_model_id == 7 else None,
    )

    config = get_model_config("trained_model_7")

    assert config["model_id"] == "trained_model_7"
    assert config["display_name"] == "Random Forest - experimento #3"
    assert config["model_family"] == "machine_learning"
    assert config["feature_mode"] == "combined"
    assert config["artifact_path"] == tmp_path / "model.joblib"
    assert config["enabled"] is True


def test_get_predictor_creates_predictor_for_registered_ml_model(monkeypatch, tmp_path):
    record = _trained_model_record(tmp_path)
    monkeypatch.setattr(
        predictors,
        "get_trained_model",
        lambda trained_model_id: record if trained_model_id == 7 else None,
    )

    get_predictor.cache_clear()
    try:
        predictor = get_predictor("trained_model_7")
    finally:
        get_predictor.cache_clear()

    assert isinstance(predictor, MLPredictor)
    assert predictor.model_path == tmp_path / "model.joblib"
    assert predictor.feature_columns_path == tmp_path / "feature_columns.json"


def test_get_model_config_rejects_missing_registered_model(monkeypatch):
    monkeypatch.setattr(predictors, "get_trained_model", lambda trained_model_id: None)

    with pytest.raises(ValueError, match="Modelo no encontrado"):
        get_model_config("trained_model_999")


def test_ml_predictor_info_uses_artifact_metadata(monkeypatch):
    predictor = _ml_predictor(monkeypatch)

    info = predictor.info()

    assert info["model_id"] == "ml_best"
    assert info["model_name"] == "fake_rf"
    assert info["n_features"] == 1
    assert info["metrics"]["cv_metrics"]["f1"] == 1.0


def test_ml_predictor_predict_aggregates_epoch_votes(monkeypatch):
    predictor = _ml_predictor(monkeypatch)
    monkeypatch.setattr(
        predictors,
        "prepare_features_from_dataframe",
        lambda df, metadata, feature_columns: (
            pd.DataFrame({"f1": [0.1, 0.2, 0.3]}),
            None,
            None,
            None,
        ),
    )

    result = predictor.predict(pd.DataFrame({"Fp1": [1.0, 2.0, 3.0]}))

    assert result["prediction_label"] == "ADHD"
    assert result["epoch_count_by_class"] == {"Control": 1, "ADHD": 2}
    assert result["confidence"] == pytest.approx(0.8)


def test_dl_predictor_info_and_predict(monkeypatch):
    predictor = _dl_predictor(monkeypatch)
    monkeypatch.setattr(
        predictors,
        "prepare_dl_epochs_from_dataframe",
        lambda df, metadata: (
            np.zeros((3, 2, 1), dtype=np.float32),
            np.array([0, 1, 1]),
            np.array(["p1", "p1", "p1"]),
        ),
    )

    info = predictor.info()
    result = predictor.predict(pd.DataFrame({"Fp1": [1.0, 2.0, 3.0]}))

    assert info["model_name"] == "fake_cnn"
    assert result["prediction_label"] == "ADHD"
    assert result["threshold"] == 0.5
    assert result["epoch_percentage_by_class"]["ADHD"] == pytest.approx(2 / 3)