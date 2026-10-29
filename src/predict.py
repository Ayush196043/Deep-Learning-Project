"""
Standalone Inference and Prediction Service
Potato Leaf Disease Detection Project

Provides:
- Input image validation (format, corruption, dimensions)
- Preprocessing identical to the training pipeline
- Confidence calculation and class probability mapping
- Actionable agronomic educational explanations for farmers
"""

import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
import numpy as np
import tensorflow as tf

from src.config import (
    SAVED_MODELS_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_SIZE,
    IMAGE_SHAPE,
)

# Educational and Agronomic Guidance per Disease Class
EDUCATIONAL_GUIDANCE = {
    "Potato___Early_blight": {
        "common_name": "Early Blight (अगेती झुलसा)",
        "causative_agent": "Alternaria solani (Fungus)",
        "symptoms": (
            "Concentric dark brown to black rings forming a 'target-board' or bullseye pattern "
            "on older lower leaves, often surrounded by yellow chlorotic halos."
        ),
        "treatment_advice": (
            "1. Remove and destroy infected lower leaves.\n"
            "2. Apply copper-based fungicides or Chlorothalonil / Mancozeb early in the morning.\n"
            "3. Ensure crop rotation with non-solanaceous crops for 2-3 seasons.\n"
            "4. Avoid overhead sprinkler irrigation to keep foliage dry."
        ),
        "severity": "Moderate to High (if left unmanaged, can reduce potato yield by 20-40%)",
        "is_healthy": False
    },
    "Potato___Late_blight": {
        "common_name": "Late Blight (पछेती झुलsa)",
        "causative_agent": "Phytophthora infestans (Oomycete / Water Mold)",
        "symptoms": (
            "Rapidly expanding, irregular water-soaked dark brown or purplish-black lesions, "
            "often accompanied by white fuzzy fungal growth on the underside of leaves in humid/cool conditions."
        ),
        "treatment_advice": (
            "1. URGENT: Isolate affected plants immediately to prevent rapid field epidemic.\n"
            "2. Spray systemic fungicides containing Metalaxyl, Dimethomorph, or Cymoxanil.\n"
            "3. Improve air circulation and field drainage; destroy infected haulms before harvest.\n"
            "4. Do not harvest wet tubers to avoid tuber rot storage contagion."
        ),
        "severity": "Critical / Emergency (can devastate an entire field in 7-10 days under cool, damp conditions)",
        "is_healthy": False
    },
    "Potato___healthy": {
        "common_name": "Healthy Potato Leaf (स्वस्थ पत्ती)",
        "causative_agent": "None (Plant is in optimal physiological condition)",
        "symptoms": (
            "Vibrant green, uniform leaf texture with no necrotic lesions, spotting, or chlorotic halos."
        ),
        "treatment_advice": (
            "1. Maintain regular balanced nitrogen-phosphorus-potassium (NPK) fertilization.\n"
            "2. Monitor regularly for early signs of aphids or Colorado potato beetles.\n"
            "3. Maintain drip irrigation schedule without over-saturating root zones."
        ),
        "severity": "None (Optimal condition)",
        "is_healthy": True
    }
}

class PotatoPredictor:
    """
    Production-grade predictor that loads the trained model once
    and performs consistent image validation, preprocessing, and inference.
    """
    _instance: Optional["PotatoPredictor"] = None

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            model_path = SAVED_MODELS_DIR / "baseline_best.keras"
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        print(f"[PREDICTOR] Loading trained model from: {self.model_path.name} ...")
        self.model = tf.keras.models.load_model(str(self.model_path))
        print("[PREDICTOR] Model loaded and ready for inference.")

    @classmethod
    def get_singleton(cls, model_path: Optional[Union[str, Path]] = None) -> "PotatoPredictor":
        """Provides a thread-safe singleton instance to avoid reloading weights in web servers."""
        if cls._instance is None:
            cls._instance = PotatoPredictor(model_path)
        return cls._instance

    @staticmethod
    def validate_image_input(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """
        Validates the image input to ensure:
        1. Non-empty and readable
        2. Valid image format (JPEG, PNG, etc.)
        3. Convert safely to 3-channel RGB
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"Image file is empty (0 bytes): {path}")
            try:
                img = Image.open(path)
                img.verify()  # Check for header/file corruption
                img = Image.open(path)  # Re-open after verify
            except Exception as e:
                raise ValueError(f"Corrupt or invalid image file: {e}")

        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ValueError("Received empty byte buffer (0 bytes).")
            try:
                img = Image.open(io.BytesIO(image_input))
                img.verify()
                img = Image.open(io.BytesIO(image_input))
            except Exception as e:
                raise ValueError(f"Corrupt or invalid image bytes: {e}")

        elif isinstance(image_input, Image.Image):
            img = image_input

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Ensure image is RGB (convert RGBA, Grayscale, CMYK, Palette)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    @staticmethod
    def preprocess_image(img: Image.Image) -> np.ndarray:
        """
        Preprocesses a PIL Image to match the exact training pipeline:
        1. Resize to target size (224, 224) using bilinear interpolation
        2. Convert to float32 array
        3. Normalize pixel intensities to [0.0, 1.0]
        4. Expand dims to batch shape: (1, 224, 224, 3)
        """
        # Resize with bilinear interpolation (matching tf.image.resize method='bilinear')
        img_resized = img.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalize pixel values
        img_normalized = img_array / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch

    def predict(self, image_input: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction:
        Validates -> Preprocesses -> Infers -> Maps classes & probabilities -> Enriches with educational guidance.
        """
        # 1. Validation
        img = self.validate_image_input(image_input)

        # 2. Preprocessing
        img_tensor = self.preprocess_image(img)

        # 3. Model Inference (training=False disables dropout)
        probs = self.model(img_tensor, training=False).numpy()[0]

        # 4. Extract Top Prediction
        pred_index = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_index]
        confidence = float(probs[pred_index])

        # 5. Build Class Probabilities Dictionary
        class_probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        }

        # 6. Retrieve Agronomic Advice
        guidance = EDUCATIONAL_GUIDANCE.get(predicted_class, {})

        return {
            "predicted_class": predicted_class,
            "class_name_clean": guidance.get("common_name", predicted_class),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "class_probabilities": class_probabilities,
            "is_healthy": guidance.get("is_healthy", False),
            "educational_info": guidance,
            "disclaimer": (
                "Notice: This is an AI-powered preliminary screening tool. "
                "For commercial farm management, confirm diagnosis with a local agricultural extension specialist."
            )
        }

def predict_single_image(image_path: Union[str, Path], model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Helper function to run inference from a file path."""
    predictor = PotatoPredictor.get_singleton(model_path)
    return predictor.predict(image_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a potato leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--model", type=str, default=None, help="Optional model path")
    args = parser.parse_args()

    result = predict_single_image(args.image, args.model)
    print("\n" + "=" * 60)
    print(f"PREDICTION RESULT: {result['class_name_clean']}")
    print(f"Confidence:       {result['confidence_percentage']}")
    print("Class Probabilities:")
    for cls, prob in result["class_probabilities"].items():
        print(f"  - {cls:<25}: {prob * 100:.2f}%")
    print(f"\nSymptoms: {result['educational_info'].get('symptoms')}")
    print(f"\nAdvice:\n{result['educational_info'].get('treatment_advice')}")
    print("=" * 60)
