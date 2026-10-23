"""Unit tests for UncertaintyEngine, Monte Carlo Dropout, and Shannon entropy."""

import numpy as np
from src.model_loader import MockKerasModel
from src.uncertainty_engine import UncertaintyEngine


def test_evaluate_uncertainty_structure(mock_model):
    engine = UncertaintyEngine(num_samples=10)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = engine.evaluate_uncertainty(mock_model, dummy_input, num_samples=10)

    assert "num_mc_samples" in res
    assert res["num_mc_samples"] == 10
    assert "predicted_class" in res
    assert "confidence_percentage" in res
    assert "shannon_entropy_bits" in res
    assert "normalized_entropy" in res
    assert "mutual_information" in res
    assert "uncertainty_tier" in res
    assert "reliability_score" in res
    assert "class_breakdown" in res
    assert len(res["class_breakdown"]) == 4


def test_shannon_entropy_bounds(mock_model):
    engine = UncertaintyEngine(num_samples=8)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = engine.evaluate_uncertainty(mock_model, dummy_input)

    # Max entropy for 4 classes is 2.0 bits
    assert 0.0 <= res["shannon_entropy_bits"] <= 2.05
    assert 0.0 <= res["normalized_entropy"] <= 1.0


def test_mutual_information_non_negative(mock_model):
    engine = UncertaintyEngine(num_samples=10)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = engine.evaluate_uncertainty(mock_model, dummy_input)

    assert res["mutual_information"] >= 0.0


def test_credible_intervals_ordering(mock_model):
    engine = UncertaintyEngine(num_samples=12)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = engine.evaluate_uncertainty(mock_model, dummy_input)

    for cls_name, meta in res["class_breakdown"].items():
        assert meta["ci_95_low"] <= meta["mean_probability"] <= meta["ci_95_high"]
        assert 0.0 <= meta["ci_95_low"]
        assert meta["ci_95_high"] <= 1.0


def test_reliability_score_range(mock_model):
    engine = UncertaintyEngine(num_samples=5)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = engine.evaluate_uncertainty(mock_model, dummy_input)

    assert 0.0 <= res["reliability_score"] <= 100.0
