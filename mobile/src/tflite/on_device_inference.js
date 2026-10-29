/**
 * On-Device TensorFlow Lite Inference Architecture Guide
 * Potato Leaf Disease Mobile Application
 *
 * Stack:
 * - Runtime: react-native-fast-tflite (C++ JSI bindings to TensorFlow Lite C++ runtime)
 * - Model: models/tflite/potato_model_dynamic_quant.tflite (428 KB) or potato_model_int8.tflite (433 KB)
 *
 * Implementation Walkthrough:
 * 1. Model Asset Bundle:
 *    The quantized TFLite file is copied to Android assets:
 *    `android/app/src/main/assets/potato_model_int8.tflite`
 *    and iOS bundle resources:
 *    `ios/PotatoApp/potato_model_int8.tflite`
 *
 * 2. JSI Loader:
 *    ```javascript
 *    import { loadTensorflowModel } from 'react-native-fast-tflite';
 *
 *    // Zero-overhead C++ pointer to neural network graph
 *    const model = await loadTensorflowModel(
 *      require('../../assets/potato_model_int8.tflite'),
 *      'default' // or 'gpu', 'nnapi'
 *    );
 *    ```
 *
 * 3. Fast Zero-Copy Inference:
 *    ```javascript
 *    // Input: Float32Array or Uint8Array of shape (1, 224, 224, 3)
 *    const output = await model.run([inputTensorBuffer]);
 *    // output[0] contains class probabilities [EarlyBlight, LateBlight, Healthy]
 *    ```
 */

export const ON_DEVICE_TFLITE_INFO = {
  model_name: "potato_model_int8.tflite",
  model_size_kb: 433.8,
  expected_latency_ms: "9 - 15 ms (NPU/CPU via XNNPACK)",
  memory_footprint: "< 15 MB RAM",
  offline_capability: "100% Autonomous (Zero cellular or Wi-Fi connectivity required)"
};
