"""
Unit Tests for Standalone Predictor and Preprocessing Consistency
Potato Leaf Disease Detection Project

Verifies:
1. End-to-end inference accuracy and output schema
2. Input validation and error handling (corrupt, non-existent, invalid formats)
3. Zero training-serving skew (preprocessing consistency between tf.data and PIL predictor)
"""

import sys
import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.config import CLASS_NAMES, NUM_CLASSES, RAW_DATA_DIR
from src.predict import PotatoPredictor, predict_single_image
from src.dataset import parse_and_preprocess_image

@pytest.fixture(scope="module")
def predictor():
    """Shared predictor instance across tests."""
    return PotatoPredictor.get_singleton()

@pytest.fixture
def sample_image_path():
    """Finds a valid sample image from raw data."""
    cls_dir = RAW_DATA_DIR / "Potato___Early_blight"
    images = list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg"))
    assert len(images) > 0, "No sample images found in raw data."
    return images[0]

def test_prediction_output_schema(predictor, sample_image_path):
    """Verify prediction dictionary structure, types, and probability constraints."""
    result = predictor.predict(sample_image_path)
    
    assert "predicted_class" in result
    assert result["predicted_class"] in CLASS_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "confidence_percentage" in result
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == NUM_CLASSES
    
    # Check probabilities sum to 1.0 within float rounding tolerance
    prob_sum = sum(result["class_probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01
    
    # Educational info checks
    assert "educational_info" in result
    assert "treatment_advice" in result["educational_info"]
    assert "symptoms" in result["educational_info"]

def test_prediction_from_bytes(predictor, sample_image_path):
    """Verify inference accepts raw image bytes (as received in HTTP multipart uploads)."""
    with open(sample_image_path, "rb") as f:
        img_bytes = f.read()
        
    result = predictor.predict(img_bytes)
    assert result["predicted_class"] in CLASS_NAMES
    assert result["confidence"] > 0.0

def test_invalid_image_inputs(predictor):
    """Verify robust exception handling for bad inputs."""
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        predictor.predict("non_existent_leaf_image_12345.jpg")
        
    # Empty byte buffer
    with pytest.raises(ValueError):
        predictor.predict(b"")
        
    # Corrupt byte buffer
    with pytest.raises(ValueError):
        predictor.predict(b"NOT_A_REAL_IMAGE_CORRUPT_BYTES")

def test_rgba_to_rgb_conversion(predictor):
    """Verify RGBA (with alpha channel) images are converted cleanly to 3-channel RGB."""
    rgba_img = Image.new("RGBA", (300, 300), color=(100, 150, 200, 128))
    validated = predictor.validate_image_input(rgba_img)
    assert validated.mode == "RGB"
    
    # Preprocess should return (1, 224, 224, 3)
    tensor = predictor.preprocess_image(validated)
    assert tensor.shape == (1, 224, 224, 3)

def test_preprocessing_consistency_no_skew(sample_image_path):
    """
    CRITICAL: Verify training pipeline and inference preprocessing are mathematically consistent.
    Compares tf.data parse_and_preprocess_image vs PotatoPredictor.preprocess_image.
    """
    # 1. Pipeline path (tf.data)
    tf_img, _ = parse_and_preprocess_image(str(sample_image_path), tf.constant(0))
    tf_array = tf_img.numpy()  # Shape: (224, 224, 3), range [0.0, 1.0]
    
    # 2. Inference path (PIL)
    pil_img = Image.open(sample_image_path).convert("RGB")
    pil_tensor = PotatoPredictor.preprocess_image(pil_img)[0]  # Squeeze batch dim -> (224, 224, 3)
    
    # Both arrays must match with high precision (mean absolute error < 0.02 due to subtle resize kernel differences)
    mae = np.mean(np.abs(tf_array - pil_tensor))
    assert mae < 0.02, f"Training-serving preprocessing skew detected! MAE={mae:.4f}"
