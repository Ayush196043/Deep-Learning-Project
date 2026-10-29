"""
Automated Integration Tests for FastAPI Backend Endpoints
Potato Leaf Disease Detection Project

Tests FastAPI endpoints directly:
- GET / (Welcome)
- GET /health (Readiness check)
- GET /classes (Class taxonomy)
- POST /predict (Valid image, corrupt image, bad mime type, empty file, oversized payload)
"""

import sys
import io
from pathlib import Path
import pytest
from fastapi import UploadFile, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import (
    app,
    root,
    health_check,
    get_supported_classes,
    predict_leaf_disease,
    MAX_FILE_SIZE_BYTES
)
from src.config import CLASS_NAMES, RAW_DATA_DIR
from src.predict import PotatoPredictor

@pytest.fixture(scope="module", autouse=True)
def preload_predictor():
    """Ensure singleton predictor is loaded in app.state for all tests."""
    app.state.predictor = PotatoPredictor.get_singleton()
    return app.state.predictor

@pytest.fixture
def sample_image():
    """Provides raw bytes of an actual leaf image."""
    cls_dir = RAW_DATA_DIR / "Potato___Early_blight"
    images = list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.jpg"))
    assert len(images) > 0, "No raw images found for API test."
    with open(images[0], "rb") as f:
        return f.read(), images[0].name

def test_root_endpoint():
    """Verify root GET endpoint returns welcome schema."""
    data = root()
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["health_check"] == "/health"

def test_health_check_endpoint():
    """Verify health and model readiness endpoint."""
    res = health_check()
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.model_name == "baseline_best.keras"

def test_classes_endpoint():
    """Verify target classes listing."""
    res = get_supported_classes()
    assert res.count == len(CLASS_NAMES)
    assert res.classes == CLASS_NAMES

def test_predict_endpoint_valid_image(sample_image):
    """Verify successful disease prediction from a valid leaf upload."""
    img_bytes, filename = sample_image
    upload_file = UploadFile(
        file=io.BytesIO(img_bytes),
        filename=filename,
        headers={"content-type": "image/jpeg"}
    )
    
    response = predict_leaf_disease(file=upload_file)
    assert response.predicted_class in CLASS_NAMES
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.class_probabilities) == len(CLASS_NAMES)
    assert response.educational_info.symptoms is not None
    assert response.educational_info.treatment_advice is not None

def test_predict_endpoint_unsupported_mime():
    """Verify rejection of non-image MIME types."""
    upload_file = UploadFile(
        file=io.BytesIO(b"Hello text"),
        filename="notes.txt",
        headers={"content-type": "text/plain"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_leaf_disease(file=upload_file)
    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in exc_info.value.detail

def test_predict_endpoint_empty_file():
    """Verify rejection of zero-byte file uploads."""
    upload_file = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.jpg",
        headers={"content-type": "image/jpeg"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_leaf_disease(file=upload_file)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()

def test_predict_endpoint_corrupt_image():
    """Verify rejection of corrupt image data pretending to be JPEG."""
    upload_file = UploadFile(
        file=io.BytesIO(b"NOT_REAL_JPEG_DATA_12345"),
        filename="fake.jpg",
        headers={"content-type": "image/jpeg"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_leaf_disease(file=upload_file)
    assert exc_info.value.status_code == 400
    assert "decoding failed" in exc_info.value.detail.lower()

def test_predict_endpoint_oversized_file():
    """Verify 413 error when file exceeds 10MB limit."""
    oversized_data = b"x" * (MAX_FILE_SIZE_BYTES + 1024)
    upload_file = UploadFile(
        file=io.BytesIO(oversized_data),
        filename="huge.jpg",
        headers={"content-type": "image/jpeg"}
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_leaf_disease(file=upload_file)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum allowed limit" in exc_info.value.detail
