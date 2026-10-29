"""
Configuration and Hyperparameters
Potato Leaf Disease Detection Project
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
TFLITE_MODELS_DIR = MODELS_DIR / "tflite"

# Dataset Specification
TARGET_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]
CLASS_NAMES = TARGET_CLASSES
NUM_CLASSES = len(TARGET_CLASSES)

# Model & Preprocessing Hyperparameters
IMAGE_SIZE = (224, 224)      # Standard input resolution
IMAGE_SHAPE = (224, 224, 3)   # (Height, Width, Channels)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Training Splits
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Baseline Training Hyperparameters
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
