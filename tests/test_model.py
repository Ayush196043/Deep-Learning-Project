"""
Unit Tests for Deep Learning Architectures
Tests Baseline CNN and MobileNetV2 transfer learning model definitions and forward passes.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.config import IMAGE_SHAPE, NUM_CLASSES
from src.model import build_baseline_cnn, build_mobilenetv2_model

def test_baseline_cnn_architecture():
    """Verify Baseline CNN input/output shapes and layer configurations."""
    model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, NUM_CLASSES)
    assert model.name == "baseline_potato_cnn"
    
    # Layer type checks
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert "Conv2D" in layer_types
    assert "MaxPooling2D" in layer_types
    assert "GlobalAveragePooling2D" in layer_types
    assert "Dropout" in layer_types
    assert "Dense" in layer_types

def test_baseline_cnn_forward_pass():
    """Verify forward pass on dummy batch and valid probability distribution."""
    model = build_baseline_cnn(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES)
    
    dummy_input = tf.random.uniform(shape=(2, 224, 224, 3), minval=0.0, maxval=1.0)
    predictions = model(dummy_input, training=False)
    
    assert predictions.shape == (2, NUM_CLASSES)
    # Check probabilities sum to 1.0 (within float tolerance)
    probs_sum = tf.reduce_sum(predictions, axis=1).numpy()
    np.testing.assert_allclose(probs_sum, [1.0, 1.0], atol=1e-5)
    # Ensure non-negative
    assert np.all(predictions.numpy() >= 0.0)

def test_mobilenetv2_architecture():
    """Verify MobileNetV2 transfer learning model structure (weights=None for isolated unit testing)."""
    model = build_mobilenetv2_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, weights=None)
    
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, NUM_CLASSES)
    
    # Check that base MobileNetV2 has non-trainable weights
    base_layers = [l for l in model.layers if "mobilenetv2" in l.name.lower()]
    assert len(base_layers) > 0
    assert base_layers[0].trainable is False

def test_mobilenetv2_forward_pass():
    """Verify MobileNetV2 forward pass and output bounds."""
    model = build_mobilenetv2_model(input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, weights=None)
    
    dummy_input = tf.random.uniform(shape=(2, 224, 224, 3), minval=0.0, maxval=1.0)
    predictions = model(dummy_input, training=False)
    
    assert predictions.shape == (2, NUM_CLASSES)
    probs_sum = tf.reduce_sum(predictions, axis=1).numpy()
    np.testing.assert_allclose(probs_sum, [1.0, 1.0], atol=1e-5)
