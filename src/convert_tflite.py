"""
TensorFlow Lite Conversion and Post-Training Quantization
Potato Leaf Disease Detection Project

Converts trained model to:
1. Standard TFLite (Float32)
2. Float16 Quantization (GPU/DSP accelerated)
3. Dynamic-Range Quantization (Integer weights, Float32 activations)
4. Full Integer Quantization (Int8 with representative dataset)

Performs:
- Size comparison (Disk footprint & compression ratio)
- Prediction consistency check (MAE against original Keras model)
- Latency benchmarks (milliseconds per image on CPU)
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    SAVED_MODELS_DIR,
    TFLITE_MODELS_DIR,
    PROCESSED_DATA_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
)

# Redirect Matplotlib cache
os.environ["MPLCONFIGDIR"] = str(PROCESSED_DATA_DIR / ".mplconfig")

import numpy as np
import tensorflow as tf
from PIL import Image

def representative_dataset_generator():
    """Yields small sample of normalized training images for INT8 calibration."""
    manifest_path = PROCESSED_DATA_DIR / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_items = [item for item in manifest if item["split"] == "train"][:100]

    for item in train_items:
        img_path = PROJECT_ROOT / item["file_path"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            yield [np.expand_dims(img_arr, axis=0)]

def convert_models() -> Dict[str, Path]:
    """Converts the best Keras model to multiple TFLite quantization targets."""
    TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    keras_model_path = SAVED_MODELS_DIR / "baseline_best.keras"
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {keras_model_path}")

    print("=" * 65)
    print(f"[TFLITE] Loading Keras model: {keras_model_path.name}")
    keras_model = tf.keras.models.load_model(str(keras_model_path))

    converted_paths = {}

    # 1. Standard Float32 TFLite
    print("\n1. Converting to Standard Float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_fp32 = converter.convert()
    fp32_path = TFLITE_MODELS_DIR / "potato_model_fp32.tflite"
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    converted_paths["Float32"] = fp32_path
    print(f"   [OK] Saved: {fp32_path.name} ({len(tflite_fp32) / 1024:.1f} KB)")

    # 2. Float16 Quantization
    print("\n2. Converting with Float16 Quantization (FP16 weights)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    fp16_path = TFLITE_MODELS_DIR / "potato_model_fp16.tflite"
    with open(fp16_path, "wb") as f:
        f.write(tflite_fp16)
    converted_paths["Float16"] = fp16_path
    print(f"   [OK] Saved: {fp16_path.name} ({len(tflite_fp16) / 1024:.1f} KB)")

    # 3. Dynamic Range Quantization (Int8 weights, Float32 activations)
    print("\n3. Converting with Dynamic Range Quantization (Int8 weights)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    dynamic_path = TFLITE_MODELS_DIR / "potato_model_dynamic_quant.tflite"
    with open(dynamic_path, "wb") as f:
        f.write(tflite_dynamic)
    converted_paths["Dynamic_Int8"] = dynamic_path
    print(f"   [OK] Saved: {dynamic_path.name} ({len(tflite_dynamic) / 1024:.1f} KB)")

    # 4. Full Integer Quantization (Representative Dataset Calibration)
    print("\n4. Converting with Full Integer (INT8) Quantization...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_generator
        tflite_int8 = converter.convert()
        int8_path = TFLITE_MODELS_DIR / "potato_model_int8.tflite"
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        converted_paths["Full_Int8"] = int8_path
        print(f"   [OK] Saved: {int8_path.name} ({len(tflite_int8) / 1024:.1f} KB)")
    except Exception as e:
        print(f"   [WARN] Full Integer Quantization encountered: {e}")

    return converted_paths

def run_tflite_inference(tflite_model_path: Path, input_tensor: np.ndarray) -> np.ndarray:
    """Executes inference using the TensorFlow Lite C++ Interpreter."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Handle quantization scale/zero-point if input tensor is int8/uint8
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        tensor_to_feed = np.round(input_tensor / scale + zero_point).astype(np.int8)
    elif input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        tensor_to_feed = np.round(input_tensor / scale + zero_point).astype(np.uint8)
    else:
        tensor_to_feed = input_tensor.astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], tensor_to_feed)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]["index"])

    # De-quantize if output is integer
    output_dtype = output_details[0]["dtype"]
    if output_dtype in (np.int8, np.uint8):
        scale, zero_point = output_details[0]["quantization"]
        return (raw_output.astype(np.float32) - zero_point) * scale
    return raw_output

def benchmark_models(converted_paths: Dict[str, Path]) -> Dict[str, Any]:
    """Compares model sizes, inference latencies, and prediction consistency."""
    keras_path = SAVED_MODELS_DIR / "baseline_best.keras"
    keras_model = tf.keras.models.load_model(str(keras_path))
    keras_size_kb = keras_path.stat().st_size / 1024

    # Create dummy test input batch of 1 image
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)

    # 1. Benchmark Keras model
    keras_probs = keras_model(dummy_input, training=False).numpy()
    
    # Warmup
    for _ in range(5):
        _ = keras_model(dummy_input, training=False)
        
    start_time = time.perf_counter()
    iterations = 30
    for _ in range(iterations):
        _ = keras_model(dummy_input, training=False)
    keras_latency_ms = ((time.perf_counter() - start_time) / iterations) * 1000

    results = {
        "Original_Keras": {
            "size_kb": round(keras_size_kb, 1),
            "size_mb": round(keras_size_kb / 1024, 2),
            "latency_ms": round(keras_latency_ms, 2),
            "mae_vs_keras": 0.0,
            "compression_ratio": "1.0x"
        }
    }

    print("\n" + "=" * 75)
    print(f"{'Model Format':<18} | {'Size (KB)':<10} | {'Latency (ms)':<12} | {'Prob MAE':<10} | {'Compression'}")
    print("-" * 75)
    print(f"{'Original Keras':<18} | {keras_size_kb:<10.1f} | {keras_latency_ms:<12.2f} | {'Baseline':<10} | {'1.0x'}")

    for name, path in converted_paths.items():
        size_kb = path.stat().st_size / 1024
        compression = f"{keras_size_kb / size_kb:.1f}x"

        # Warmup TFLite
        for _ in range(5):
            _ = run_tflite_inference(path, dummy_input)

        # Measure Latency
        start_t = time.perf_counter()
        for _ in range(iterations):
            tflite_out = run_tflite_inference(path, dummy_input)
        latency_ms = ((time.perf_counter() - start_t) / iterations) * 1000

        # Calculate Mean Absolute Error against Keras baseline probabilities
        mae = float(np.mean(np.abs(keras_probs - tflite_out)))

        results[name] = {
            "size_kb": round(size_kb, 1),
            "size_mb": round(size_kb / 1024, 2),
            "latency_ms": round(latency_ms, 2),
            "mae_vs_keras": round(mae, 6),
            "compression_ratio": compression,
            "filename": path.name
        }

        print(f"{name:<18} | {size_kb:<10.1f} | {latency_ms:<12.2f} | {mae:<10.6f} | {compression}")

    print("=" * 75)

    # Save benchmark json
    summary_path = PROCESSED_DATA_DIR / "tflite_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Benchmark summary persisted to: {summary_path}")

    return results

if __name__ == "__main__":
    paths = convert_models()
    benchmark_models(paths)
