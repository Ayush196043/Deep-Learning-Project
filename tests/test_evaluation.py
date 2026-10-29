"""
Unit Tests for Evaluation Pipeline
Potato Leaf Disease Detection Project

Tests confusion matrix computation, precision/recall/F1 metrics, and evaluation artifact schemas.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_confusion_matrix, compute_classification_metrics

def test_confusion_matrix_computation():
    """Verify confusion matrix counting for known synthetic labels."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    
    # Expected:
    # Class 0: 1 correct (0->0), 1 predicted as 1 (0->1)
    # Class 1: 2 correct (1->1)
    # Class 2: 1 correct (2->2), 1 predicted as 0 (2->0)
    expected_cm = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1]
    ])
    np.testing.assert_array_equal(cm, expected_cm)

def test_classification_metrics_computation():
    """Verify Precision, Recall, and F1 calculations from a known confusion matrix."""
    cm = np.array([
        [10, 2, 0],
        [1, 15, 0],
        [0, 1, 8]
    ])
    class_names = ["Early_blight", "Late_blight", "healthy"]
    
    metrics = compute_classification_metrics(cm, class_names)
    
    assert metrics["total_test_samples"] == 37
    assert metrics["accuracy"] == round(33 / 37, 4)
    
    # Class 0 (Early_blight): TP=10, FP=1, FN=2
    # Precision = 10 / 11 = 0.9091
    # Recall = 10 / 12 = 0.8333
    cls0 = metrics["per_class"]["Early_blight"]
    assert cls0["precision"] == round(10 / 11, 4)
    assert cls0["recall"] == round(10 / 12, 4)
    assert cls0["support"] == 12

    # Check macro averages are averages of per-class
    p_sum = sum(metrics["per_class"][c]["precision"] for c in class_names)
    assert abs(metrics["macro_avg"]["precision"] - (p_sum / 3)) < 1e-3

def test_metrics_schema_and_ranges():
    """Verify all metrics fall within valid [0.0, 1.0] mathematical boundaries."""
    cm = np.array([
        [80, 15, 5],
        [10, 85, 5],
        [2, 3, 11]
    ])
    class_names = ["A", "B", "C"]
    metrics = compute_classification_metrics(cm, class_names)
    
    assert 0.0 <= metrics["accuracy"] <= 1.0
    for c in class_names:
        assert 0.0 <= metrics["per_class"][c]["precision"] <= 1.0
        assert 0.0 <= metrics["per_class"][c]["recall"] <= 1.0
        assert 0.0 <= metrics["per_class"][c]["f1_score"] <= 1.0
    
    assert 0.0 <= metrics["macro_avg"]["precision"] <= 1.0
    assert 0.0 <= metrics["macro_avg"]["recall"] <= 1.0
    assert 0.0 <= metrics["macro_avg"]["f1_score"] <= 1.0
