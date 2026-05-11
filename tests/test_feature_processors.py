"""Tests for feature processors."""

import pytest
from src.features.factory import FeatureFactory

def test_create_manual_processor():
    """Test creation of manual feature processor."""
    config = {}
    processor = FeatureFactory.get_processor("manual", config)
    assert processor is not None

def test_create_sequential_processor():
    """Test creation of sequential processor."""
    config = {}
    processor = FeatureFactory.get_processor("sequential", config)
    assert processor is not None

def test_invalid_processor():
    """Test error on invalid processor name."""
    with pytest.raises(ValueError):
        FeatureFactory.get_processor("invalid_processor", {})
