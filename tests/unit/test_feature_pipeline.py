import numpy as np
import pandas as pd
import pytest

from scripts.feature_pipeline import (
    align_feature_columns,
    build_features_from_config,
    build_features_from_epochs,
    normalize_feature_mode,
)


def test_normalize_feature_mode_accepts_time_alias():
    assert normalize_feature_mode("time") == "temporal"
    assert normalize_feature_mode("temporal") == "temporal"


def test_build_features_from_epochs_temporal_mode():
    x_epochs = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    features = build_features_from_epochs(
        x_epochs=x_epochs,
        channel_names=["Fp1", "Fp2"],
        feature_mode="temporal",
    )

    assert "Fp1_mean" in features.columns
    assert "Fp2_range" in features.columns
    assert features.loc[0, "Fp1_mean"] == pytest.approx(3.0)


def test_build_features_from_config_combined_mode():
    x_epochs = np.random.default_rng(42).standard_normal((2, 128, 2))
    config = {"feature_mode": "combined", "sfreq": 128, "nperseg": 64}

    features = build_features_from_config(x_epochs, ["Fp1", "Fp2"], config)

    assert "Fp1_mean" in features.columns
    assert "Fp1_alpha_abs_power" in features.columns


def test_align_feature_columns_preserves_expected_order():
    features = pd.DataFrame({"b": [2], "a": [1], "c": [3]})

    aligned = align_feature_columns(features, ["a", "b"])

    assert aligned.columns.tolist() == ["a", "b"]


def test_align_feature_columns_reports_missing_features():
    features = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="Faltan caracteristicas"):
        align_feature_columns(features, ["a", "missing"])
