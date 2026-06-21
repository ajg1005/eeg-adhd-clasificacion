import numpy as np
import pandas as pd
import pytest

from backend.modeling import predictors
from backend.modeling.predictors import DLPredictor, MLPredictor, get_model_config


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


def test_get_model_config_rejects_unknown_model():
    with pytest.raises(ValueError, match="Modelo no encontrado"):
        get_model_config("missing")


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
