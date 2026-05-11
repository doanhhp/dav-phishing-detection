"""Tests for evaluation metrics."""

import numpy as np
from src.evaluation.metrics import Metrics

def test_metrics_calculation():
    """Test metrics calculation."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    metrics = Metrics.calculate_all(y_true, y_pred)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
