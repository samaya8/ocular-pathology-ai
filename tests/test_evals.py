"""Unit tests for evals runner and metrics calculation."""

from pathlib import Path
import numpy as np
from evals.runner import (
    calculate_calibration_metrics,
    calculate_multiclass_metrics,
    generate_curated_eval_cohort,
    run_eval_suite,
)


def test_multiclass_metrics_perfect_accuracy():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    y_prob = np.eye(4)

    metrics = calculate_multiclass_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["per_class"]) == 4


def test_multiclass_metrics_partial_accuracy():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    y_prob = np.zeros((8, 4))
    for i, p in enumerate(y_pred):
        y_prob[i, p] = 1.0

    metrics = calculate_multiclass_metrics(y_true, y_pred, y_prob)

    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_calibration_metrics_brier_and_ece():
    y_true = np.array([0, 1, 2, 3])
    y_prob = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.05, 0.85, 0.05, 0.05],
        [0.02, 0.08, 0.80, 0.10],
        [0.01, 0.04, 0.05, 0.90],
    ])

    cal = calculate_calibration_metrics(y_true, y_prob)

    assert "multi_class_brier" in cal
    assert "ece" in cal
    assert 0.0 <= cal["multi_class_brier"] <= 2.0
    assert 0.0 <= cal["ece"] <= 1.0


def test_generate_curated_eval_cohort():
    cohort = generate_curated_eval_cohort()

    assert len(cohort) >= 4
    for img, label, desc in cohort:
        assert label in [0, 1, 2, 3]
        assert img.size[0] > 100
        assert img.size[1] > 100


def test_run_eval_suite_mock(tmp_path: Path, mock_retina_engine):
    rep_file = tmp_path / "test_benchmark.md"
    results = run_eval_suite(report_path=rep_file, engine=mock_retina_engine)

    assert rep_file.exists()
    assert "classification_metrics" in results
    assert "calibration_metrics" in results
    assert "uncertainty_summary" in results
    assert "latency_benchmark" in results
