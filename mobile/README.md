# Potato Leaf Disease Mobile Application (React Native / Expo)

A field-ready cross-platform mobile application for Android and iOS that allows farmers and agronomists to photograph potato foliage in real-time and obtain instant disease diagnoses.

---

## 1. Quickstart & Running Instructions

### Prerequisites
- Node.js >= 18
- Expo CLI (`npm install -g expo-cli`)
- Expo Go App on Android/iOS (or Android Studio Emulator / Xcode Simulator)

### Installation & Launch
```bash
cd mobile
npm install
npx expo start
```
- Press `a` to open in Android Emulator.
- Press `i` to open in iOS Simulator.
- Or scan the terminal QR code with the **Expo Go** app on your physical smartphone!

> **Connecting to Local FastAPI Backend:**
> - In `mobile/src/services/api.js`:
>   - For Android Emulator: `http://10.0.2.2:8000` (default)
>   - For Physical Phone via Wi-Fi: Replace with your PC's local IP (e.g., `http://192.168.1.15:8000`).

---

## 2. On-Device Edge Inference vs. Server-Side Cloud Inference

In agricultural deep learning, choosing between on-device and server-side execution is a critical systems engineering trade-off:

| Evaluation Dimension | Server-Side Inference (FastAPI / Cloud Run) | On-Device Edge Inference (TFLite / Mobile NPU) |
|:---|:---|:---|
| **Network Dependency** | Requires active 4G/5G or Wi-Fi internet connection | **100% Offline (Works in remote rural farms with zero signal)** |
| **Inference Latency** | Network RTT (100-800ms) + Server inference (35ms) | **Sub-10ms (Immediate execution on local NPU/CPU)** |
| **Bandwidth Costs** | Consumes mobile data uploading full-resolution JPEG photos | **Zero cellular data consumption** |
| **Model Security & IP** | Weights are protected inside server memory | Model binary is packaged inside mobile APK/IPA |
| **Model Size Limits** | Can host large models (e.g., ResNet50, ViT, Ensemble) | Model must be aggressively quantized (e.g., INT8 < 1MB) |
| **Privacy & Data Sovereignty**| Photos leave the farmer's device | Photos remain strictly local on the smartphone |
| **Battery & Thermal Impact** | Low battery impact on phone (offloaded to cloud) | Heavy continuous inference can heat phone and drain battery |
| **Model Updates** | Instantaneous across all users (single server deploy) | Requires app store update or dynamic OTA model sync |
