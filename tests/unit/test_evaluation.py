import numpy as np
import pytest

from scripts.evaluation import find_best_threshold, metrics_dict


def test_metrics_dict_returns_standard_binary_metrics():
    metrics = metrics_dict([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_find_best_threshold_prefers_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])

    threshold = find_best_threshold(y_true, y_score, lo=0.2, hi=0.8, n_points=7)

    assert threshold == pytest.approx(0.3)
