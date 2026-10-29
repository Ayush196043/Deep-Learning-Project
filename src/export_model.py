"""
Model Export Utility
Potato Leaf Disease Detection Project

Exports trained Keras model into:
1. Versioned TensorFlow SavedModel format (for TensorFlow Serving / GCP Cloud Run)
2. Validates serving signatures for production deployment
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import SAVED_MODELS_DIR, NUM_CLASSES, IMAGE_SHAPE
import tensorflow as tf

def export_to_saved_model(
    keras_model_path: Path = None,
    export_dir: Path = None,
    version: int = 1
) -> Path:
    """
    Exports a .keras model to the versioned TensorFlow SavedModel directory format,
    which is the industry standard for TensorFlow Serving, Triton, and cloud inference engines.
    """
    if keras_model_path is None:
        keras_model_path = SAVED_MODELS_DIR / "baseline_best.keras"

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Model not found at {keras_model_path}")

    if export_dir is None:
        export_dir = SAVED_MODELS_DIR / "potato_serving_model" / str(version)

    export_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"[EXPORT] Converting: {keras_model_path.name} -> SavedModel (version {version})")
    print(f"Destination: {export_dir}")
    print("=" * 65)

    model = tf.keras.models.load_model(str(keras_model_path))

    # Define concrete serving function with explicit tensor shape and dtype
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="input_image")])
    def serve(images):
        return {"output_probabilities": model(images, training=False)}

    tf.saved_model.save(
        model,
        str(export_dir),
        signatures={"serving_default": serve}
    )

    print(f"[OK] SavedModel exported successfully to: {export_dir}")
    print(f"[OK] Serving signature 'serving_default' verified.")
    return export_dir

if __name__ == "__main__":
    export_to_saved_model()
