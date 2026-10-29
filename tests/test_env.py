"""
Environment and dependency test suite.
"""

import sys
import pytest

def test_python_version():
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"

def test_tensorflow_import():
    import tensorflow as tf
    assert tf.__version__ is not None

def test_fastapi_import():
    import fastapi
    assert fastapi.__version__ is not None

def test_numpy_and_pillow():
    import numpy as np
    from PIL import Image
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert img.size == (10, 10)

def test_config_constants():
    from src.config import TARGET_CLASSES, IMAGE_SIZE, BATCH_SIZE
    assert len(TARGET_CLASSES) == 3
    assert "Potato___Early_blight" in TARGET_CLASSES
    assert "Potato___Late_blight" in TARGET_CLASSES
    assert "Potato___healthy" in TARGET_CLASSES
    assert IMAGE_SIZE == (224, 224)
    assert BATCH_SIZE > 0
