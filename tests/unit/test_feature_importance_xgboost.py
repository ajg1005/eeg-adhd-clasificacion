import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import feature_importance_xgboost as script


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Entradas pequenas, sin acceder al dataset ni a los modelos del proyecto."""
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"channels": ["Fz", "Pz"]}), encoding="utf-8")
    columns = tmp_path / "columns.json"
    columns.write_text(json.dumps(["Fz_mean", "Pz_mean"]), encoding="utf-8")
    monkeypatch.setattr(script, "METADATA_PATH", metadata)
    monkeypatch.setattr(script, "FEATURE_COLUMNS_PATH", columns)
    monkeypatch.setattr(script, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(script, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(script, "N_REPEATS", 2)
    monkeypatch.setattr(script, "TOP_N", 2)
    rng = np.random.default_rng(42)
    train = pd.DataFrame(rng.normal(size=(24, 2)), columns=["Fz_mean", "Pz_mean"])
    test = pd.DataFrame(rng.normal(size=(8, 2)), columns=train.columns)
    y_train = np.array([0] * 16 + [1] * 8)
    y_test = np.array([0, 1] * 4)
    prepare = Mock(return_value=(train, test, y_train, y_test, None, None))
    monkeypatch.setattr(script, "prepare_importance_data", prepare)
    return train, test, y_train, y_test


def test_main_trains_and_exports_importance(experiment, monkeypatch):
    train, test, y_train, y_test = experiment
    model = script.create_ml_model("xgboost")
    model.set_params(model__n_estimators=5, model__max_depth=2)
    fit = Mock(wraps=model.fit)
    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(script, "create_ml_model", Mock(return_value=model))
    permutation = Mock(wraps=script.permutation_importance)
    monkeypatch.setattr(script, "permutation_importance", permutation)

    script.main()

    assert fit.call_args.args[0] is train
    np.testing.assert_array_equal(fit.call_args.args[1], y_train)
    weights = fit.call_args.kwargs["model__sample_weight"]
    assert weights[y_train == 0].sum() == pytest.approx(weights[y_train == 1].sum())
    pd.testing.assert_frame_equal(permutation.call_args.args[1], test)
    np.testing.assert_array_equal(permutation.call_args.args[2], y_test)
    assert permutation.call_args.kwargs["scoring"] == "f1_weighted"
    assert permutation.call_args.kwargs["n_repeats"] == 2

    features = pd.read_csv(script.RESULTS_DIR / "xgboost_feature_importance.csv")
    channels = pd.read_csv(script.RESULTS_DIR / "xgboost_feature_importance_by_channel.csv")
    assert set(features["feature"]) == set(train.columns)
    assert features["importance_mean"].is_monotonic_decreasing
    assert np.isfinite(features[["importance_mean", "importance_std"]]).all().all()
    expected = script.aggregate_by_channel(features, ["Fz", "Pz"])
    pd.testing.assert_frame_equal(channels, expected, atol=1e-12)
    for name in ("xgboost_feature_importance_top2.png", "xgboost_feature_importance_by_channel.png"):
        with Image.open(script.FIGURES_DIR / name) as picture:
            assert picture.format == "PNG"
            assert picture.width > 100 and picture.height > 100
            picture.verify()


def test_dry_run_does_not_train_or_write_outputs(experiment, monkeypatch):
    monkeypatch.setattr(script, "DRY_RUN", True)
    create_model = Mock()
    monkeypatch.setattr(script, "create_ml_model", create_model)

    script.main()

    create_model.assert_not_called()
    assert not script.RESULTS_DIR.exists()
    assert not script.FIGURES_DIR.exists()
