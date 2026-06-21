import numpy as np
import pandas as pd
import pytest

from backend.services import training_runners
from backend.services.training_runners import (
    _aggregate_importance_by_channel,
    _importance_rows,
    _safe_feature_importance_for_fold,
    _stratified_subsample,
    patient_results,
)


def test_patient_results_aggregates_epoch_votes_by_patient():
    rows = patient_results(
        groups=np.array(["p1", "p1", "p2", "p2", "p2"]),
        y_true=np.array([0, 0, 1, 1, 1]),
        y_pred=np.array([0, 1, 1, 1, 0]),
    )

    assert rows[0]["patient_id"] == "p1"
    assert rows[0]["true_label"] == "Control"
    assert rows[0]["control_epoch_percentage"] == pytest.approx(0.5)
    assert rows[1]["patient_id"] == "p2"
    assert rows[1]["predicted_label"] == "ADHD"
    assert rows[1]["correct"] is True


def test_importance_rows_limits_and_serializes_values():
    importance_df = pd.DataFrame(
        {
            "feature": ["Fp1_mean", "Fp2_mean"],
            "importance_mean": [0.2, 0.1],
            "importance_std": [0.01, 0.02],
        }
    )

    rows = _importance_rows(importance_df, limit=1)

    assert rows == [
        {
            "feature": "Fp1_mean",
            "importance_mean": pytest.approx(0.2),
            "importance_std": pytest.approx(0.01),
        }
    ]


def test_aggregate_importance_by_channel_sums_matching_features():
    importance_df = pd.DataFrame(
        {
            "feature": ["Fp1_mean", "Fp1_std", "Fp2_mean", "C3_mean"],
            "importance_mean": [0.2, 0.3, 0.1, 0.0],
            "importance_std": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = _aggregate_importance_by_channel(importance_df, ["Fp1", "Fp2", "C3"])

    assert result.iloc[0]["feature"] == "Fp1"
    assert result.iloc[0]["importance_mean"] == pytest.approx(0.5)
    assert result.iloc[-1]["feature"] == "C3"


def test_stratified_subsample_returns_original_when_under_limit():
    X = pd.DataFrame({"a": [1, 2]})
    y = np.array([0, 1])

    x_sample, y_sample = _stratified_subsample(X, y, max_rows=5)

    assert x_sample.equals(X.reset_index(drop=True))
    assert y_sample.tolist() == [0, 1]


def test_stratified_subsample_limits_rows_when_possible():
    X = pd.DataFrame({"a": range(10)})
    y = np.array([0, 1] * 5)

    x_sample, y_sample = _stratified_subsample(X, y, max_rows=4)

    assert len(x_sample) == 4
    assert len(y_sample) == 4
    assert set(y_sample) == {0, 1}


def test_safe_feature_importance_returns_error_payload(monkeypatch):
    def fail_importance(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(training_runners, "_feature_importance_for_fold", fail_importance)

    result = _safe_feature_importance_for_fold(
        model=object(),
        X_test=pd.DataFrame({"Fp1_mean": [0.1]}),
        y_test=np.array([0]),
        eeg_columns=["Fp1"],
        fold=2,
    )

    assert result["method"] == "permutation_importance"
    assert result["evaluated_epochs"] == 0
    assert result["error"] == "boom"
