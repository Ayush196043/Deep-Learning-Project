"""
FastAPI Application Entrypoint
Potato Leaf Disease Detection Backend

Features:
- Singleton Model Loader on Lifespan Startup (no per-request reloading)
- Strict File Validation (file size limit, mime type, corrupt byte detection)
- CORS Middleware for React and React Native clients
- Structured Pydantic response models and OpenAPI docs
"""

import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import CLASS_NAMES, SAVED_MODELS_DIR
from src.predict import PotatoPredictor
from backend.app.schemas import (
    HealthResponse,
    ClassesResponse,
    PredictionResponse,
    ErrorResponse
)

# Configuration Constants
API_VERSION = "1.0.0"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Loads the Deep Learning model ONCE into memory on startup and caches it in app.state.
    Ensures sub-100ms inference response times by eliminating disk re-reads.
    """
    print("=" * 60)
    print(f"[FASTAPI] Starting Potato Disease Detection API v{API_VERSION}")
    print("[FASTAPI] Loading singleton predictor into memory...")
    try:
        predictor = PotatoPredictor.get_singleton()
        app.state.predictor = predictor
        print("[FASTAPI] Model loaded successfully. Server ready to accept requests.")
    except Exception as e:
        print(f"[FASTAPI ERROR] Failed to load model on startup: {e}")
        app.state.predictor = None
    print("=" * 60)
    yield
    print("[FASTAPI] Shutting down API service.")

# Initialize FastAPI App
app = FastAPI(
    title="Potato Leaf Disease Detection API",
    description=(
        "Production-grade Deep Learning API for detecting Early Blight, Late Blight, "
        "and Healthy potato leaves using Convolutional Neural Networks."
    ),
    version=API_VERSION,
    lifespan=lifespan
)

# Enable CORS for React Web and React Native Mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/",
    tags=["General"],
    summary="Root Welcome Endpoint"
)
def root():
    return {
        "message": "Welcome to Potato Leaf Disease Detection API",
        "version": API_VERSION,
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and Readiness Check"
)
def health_check():
    """Returns system status and model readiness for load balancers and container orchestrators."""
    is_loaded = getattr(app.state, "predictor", None) is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name="baseline_best.keras",
        version=API_VERSION
    )

@app.get(
    "/classes",
    response_model=ClassesResponse,
    tags=["Inference"],
    summary="List Supported Disease Classes"
)
def get_supported_classes():
    """Returns the list of 3 target potato disease classes."""
    return ClassesResponse(
        classes=CLASS_NAMES,
        count=len(CLASS_NAMES)
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or corrupt image"},
        413: {"model": ErrorResponse, "description": "File size exceeds 10MB limit"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Inference"],
    summary="Predict Disease from Potato Leaf Image"
)
def predict_leaf_disease(
    file: UploadFile = File(..., description="Uploaded potato leaf image (JPEG, PNG, WebP)")
):
    """
    Accepts an uploaded image file, verifies dimensions and format,
    runs CNN inference, and returns predicted disease, confidence score, and advisory treatment.
    """
    # 1. Verify Model Readiness
    predictor: PotatoPredictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load on startup."
        )

    # 2. Validate MIME Type
    content_type = file.content_type.lower() if file.content_type else ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type}'. Allowed types: {sorted(list(ALLOWED_MIME_TYPES))}"
        )

    # 3. Read File Bytes & Validate Size
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
        )

    # 4. Execute Prediction with Validation
    try:
        prediction_result = predictor.predict(contents)
        return PredictionResponse(**prediction_result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image decoding failed: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
