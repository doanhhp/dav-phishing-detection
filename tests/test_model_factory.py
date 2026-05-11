"""Tests for model factory."""

import pytest
from src.models.model_factory import ModelFactory

def test_create_hybrid_svm_knn():
    """Test creation of SVM+KNN model."""
    config = {}
    model = ModelFactory.create_model("hybrid_svm_knn", config)
    assert model is not None

def test_create_lstm_url():
    """Test creation of LSTM model."""
    config = {}
    model = ModelFactory.create_model("lstm_url", config)
    assert model is not None

def test_invalid_model():
    """Test error on invalid model name."""
    with pytest.raises(ValueError):
        ModelFactory.create_model("invalid_model", {})
