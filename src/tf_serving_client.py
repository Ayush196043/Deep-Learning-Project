"""
TensorFlow Serving REST Client
Potato Leaf Disease Detection Project

Demonstrates how to consume a production TensorFlow Serving instance via REST API
and parse response tensors into structured classification outputs.
"""

import sys
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, NUM_CLASSES
from src.predict import PotatoPredictor, EDUCATIONAL_GUIDANCE

DEFAULT_TF_SERVING_URL = "http://localhost:8501/v1/models/potato_model:predict"

def predict_via_tf_serving(
    image_input: Union[str, Path, bytes, Image.Image],
    endpoint_url: str = DEFAULT_TF_SERVING_URL,
    timeout_seconds: float = 5.0
) -> Dict[str, Any]:
    """
    Sends preprocessed image tensor to TensorFlow Serving REST endpoint.
    
    Payload format expected by TF Serving:
    {"signature_name": "serving_default", "instances": [[[[...]]]]}
    """
    # 1. Preprocess image using existing canonical pipeline (no duplicate code)
    img = PotatoPredictor.validate_image_input(image_input)
    tensor = PotatoPredictor.preprocess_image(img)  # Shape: (1, 224, 224, 3)

    # 2. Build REST request payload
    payload = {
        "signature_name": "serving_default",
        "instances": tensor.tolist()
    }

    # 3. Post to TensorFlow Serving
    try:
        response = requests.post(
            endpoint_url,
            data=json.dumps(payload),
            headers={"content-type": "application/json"},
            timeout=timeout_seconds
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to communicate with TensorFlow Serving at {endpoint_url}. "
            f"Ensure the Docker container is running: {e}"
        )

    # 4. Parse Model Probabilities
    response_data = response.json()
    predictions = response_data.get("predictions", [])
    if not predictions:
        raise ValueError(f"Empty predictions received from TF Serving: {response_data}")

    probs = np.array(predictions[0])
    pred_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])

    guidance = EDUCATIONAL_GUIDANCE.get(predicted_class, {})

    return {
        "predicted_class": predicted_class,
        "class_name_clean": guidance.get("common_name", predicted_class),
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        },
        "is_healthy": guidance.get("is_healthy", False),
        "educational_info": guidance,
        "serving_engine": "TensorFlow Serving (C++ REST API)"
    }

if __name__ == "__main__":
    print("=" * 65)
    print("TensorFlow Serving Client Helper")
    print("Docker command to launch TF Serving for this model:")
    print("-" * 65)
    print(
        'docker run -t --rm -p 8501:8501 '
        '-v "d:/Research Paper/Deep Learning project/models/saved_models/potato_serving_model:/models/potato_model" '
        '-e MODEL_NAME=potato_model tensorflow/serving'
    )
    print("=" * 65)
