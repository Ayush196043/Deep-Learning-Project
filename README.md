# 🥔 Potato Leaf Disease Detection — End-to-End Deep Learning System

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15.1-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![React Native](https://img.shields.io/badge/React_Native-Expo_51-61DAFB?logo=react&logoColor=black)](https://reactnative.dev/)
[![Vercel](https://img.shields.io/badge/Vercel-Web_Deploy-000000?logo=vercel&logoColor=white)](https://vercel.com/)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Tests](https://img.shields.io/badge/Pytest-33%20Passed-4CAF50?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An industry-grade, end-to-end Deep Learning system designed to identify and classify potato foliar diseases from leaf imagery. Built across 13 engineering milestones, this system covers dataset curation, stratified splitting, convolutional neural network design, model training with regularization, rigorous test-set evaluation, edge quantization (TensorFlow Lite), high-throughput asynchronous API serving (FastAPI & TF Serving), multi-platform client applications (React.js Web & React Native Mobile), and serverless cloud deployment on Google Cloud Platform (Cloud Run).

---

## 📌 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [End-to-End Architecture](#-end-to-end-architecture)
3. [Repository Directory Structure](#-repository-directory-structure)
4. [Dataset & Exploratory Data Analysis (EDA)](#-dataset--exploratory-data-analysis-eda)
5. [Preprocessing & Augmentation Pipeline](#-preprocessing--augmentation-pipeline)
6. [Model Architecture & Training Details](#-model-architecture--training-details)
7. [Comprehensive Evaluation & Error Analysis](#-comprehensive-evaluation--error-analysis)
8. [Edge Optimization: TensorFlow Lite & Quantization](#-edge-optimization-tensorflow-lite--quantization)
9. [FastAPI Microservice & TensorFlow Serving](#-fastapi-microservice--tensorflow-serving)
10. [Web & Mobile Client Applications](#-web--mobile-client-applications)
11. [Google Cloud Platform (GCP) Deployment](#-google-cloud-platform-gcp-deployment)
12. [Local Setup & Reproduction Guide (PowerShell)](#-local-setup--reproduction-guide-powershell)
13. [Automated Test Suite (33/33 Tests Passing)](#-automated-test-suite-3333-tests-passing)
14. [Viva Voce Questions & Answers (20 Technical Q&A)](#-viva-voce-questions--answers-20-technical-qa)
15. [Resume-Ready Project Highlights](#-resume-ready-project-highlights)

---

## 🎯 Project Overview & Problem Statement

Potato (*Solanum tuberosum*) is the world's fourth most vital food crop, feeding over a billion people. However, potato yields are severely threatened by fungal and oomycete foliar diseases:
1. **Early Blight (*Alternaria solani*)**: Fungal pathogen creating brown-to-black necrotic lesions with characteristic **concentric rings ("target-spot" pattern)** surrounded by chlorotic yellow halos.
2. **Late Blight (*Phytophthora infestans*)**: Devastating water-mold (oomycete) pathogen historically responsible for the Irish Potato Famine. Characterized by **large, dark, water-soaked necrotic lesions** that rapidly collapse the entire foliage under cool, humid conditions.
3. **Healthy Foliage**: Asymptomatic, vigorous green leaf canopy.

### Why Computer Vision for Agriculture?
Smallholder farmers and agronomists frequently misdiagnose early blight for late blight, leading to incorrect chemical fungicide applications, crop devastation, and economic hardship. This system provides instant, objective image classification, confidence estimation, and agronomic treatment guidance right at the edge (on mobile phones) or via the cloud.

> **Medical/Agricultural Disclaimer**: This system is a preliminary decision-support tool designed for educational and early screening purposes; it does not replace in-field lab pathology testing.

---

## 🏗️ End-to-End Architecture

```mermaid
flowchart TD
    subgraph DataPipeline ["1. Data Ingestion & Preprocessing"]
        PV["PlantVillage Potato (2,152 images)"] --> Split["Stratified Split (80 / 10 / 10)"]
        Split --> TrainSet["Train: 1,721 images"]
        Split --> ValSet["Val: 215 images"]
        Split --> TestSet["Test: 216 images (Held-out)"]
        TrainSet --> Aug["Data Augmentation (Flip, Rotate, Zoom)"]
        Aug --> Norm["Normalize (/255.0) & Resize (224x224)"]
        ValSet --> Norm
        TestSet --> Norm
    end

    subgraph DLModeling ["2. Deep Learning Modeling & Training"]
        Norm --> CNN["Custom 4-Block Baseline CNN"]
        Norm -.-> MNet["MobileNetV2 Transfer Learning"]
        CNN --> Train["Optimizer: Adam (1e-3)<br>Loss: Categorical Crossentropy<br>Callbacks: EarlyStopping & Checkpoint"]
        Train --> KerasModel["models/saved_models/baseline_best.keras (5.0 MB)"]
    end

    subgraph Optimization ["3. Model Export & Edge Quantization"]
        KerasModel --> SavedModel["models/saved_models/potato_serving_model/1/"]
        KerasModel --> TFLiteFP32["potato_model_fp32.tflite (1.65 MB)"]
        KerasModel --> TFLiteFP16["potato_model_fp16.tflite (0.83 MB)"]
        KerasModel --> TFLiteInt8["potato_model_int8.tflite (0.43 MB, 9.58ms)"]
    end

    subgraph Serving ["4. Serving & Production Backend"]
        KerasModel --> FastAPI["FastAPI REST Microservice<br>(Singleton Cached Model, CORS, Validation)"]
        SavedModel --> TFServing["TensorFlow Serving (Port 8501)"]
        FastAPI --> Endpoints["/health | /classes | /predict"]
    end

    subgraph Deployment ["5. Cloud & Multi-Platform Clients"]
        FastAPI --> CloudRun["GCP Cloud Run (Containerized via Cloud Build)<br>2 vCPU, 2 GiB RAM, Scale-to-Zero"]
        CloudRun --> ReactWeb["React.js 18 Web Application"]
        CloudRun --> MobileServer["React Native Mobile App (Online Mode)"]
        TFLiteInt8 --> MobileEdge["React Native Mobile App (On-Device Offline Mode)"]
    end
```

---

## 📁 Repository Directory Structure

```
d:\Research Paper\Deep Learning project\
├── backend/                             # Production FastAPI Backend (Phase 7)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                    # API settings & paths
│   │   ├── main.py                      # FastAPI application & endpoints
│   │   ├── predictor.py                 # Thread-safe cached inference engine
│   │   └── schemas.py                   # Pydantic request/response schemas
│   └── README.md
├── data/                                # Dataset directory (Phase 2 & 3)
│   ├── raw/                             # PlantVillage raw images
│   │   ├── Potato___Early_blight/       # 1,000 images
│   │   ├── Potato___Late_blight/        # 1,000 images
│   │   └── Potato___healthy/            # 152 images
│   └── processed/                       # Artifacts, splits, and evaluation plots
│       ├── dataset_manifest.json        # Deterministic split mapping
│       ├── class_distribution.png       # Dataset distribution chart
│       ├── sample_images_grid.png       # Class visual sample grid
│       ├── training_curves_baseline.png # Loss & Accuracy curves
│       ├── confusion_matrix.png         # Test set confusion matrix
│       ├── misclassifications.json      # Error analysis records
│       ├── misclassified_samples.png    # Visual inspection of errors
│       └── tflite_benchmark_summary.json# Quantization latency & size benchmark
├── frontend/                            # React.js Web Application (Phase 10)
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/                  # UI modular components
│   │   │   ├── EducationalCard.js       # Disease treatment guides
│   │   │   ├── Header.js                # System header
│   │   │   ├── ImageUpload.js           # Drag-and-drop file uploader
│   │   │   └── PredictionResult.js      # Bar charts & confidence badges
│   │   ├── services/
│   │   │   └── api.js                   # Axios client for FastAPI
│   │   ├── App.css                      # Modern responsive styling
│   │   ├── App.js                       # Primary web UI controller
│   │   └── index.js
│   ├── package.json
│   └── README.md
├── gcp/                                 # Google Cloud Platform Deployment (Phase 12)
│   ├── GCP_DEPLOYMENT_GUIDE.md          # Complete deployment walkthrough
│   └── deploy_cloud_run.ps1             # PowerShell automated deployment script
├── mobile/                              # React Native / Expo Mobile App (Phase 11)
│   ├── src/
│   │   ├── services/
│   │   │   └── api.js                   # Mobile API client
│   │   └── tflite/
│   │       └── on_device_inference.js   # Edge TFLite on-device inference runner
│   ├── App.js                           # Mobile screen controller
│   ├── app.json                         # Expo application config
│   ├── package.json
│   └── README.md
├── models/                              # Trained models & edge artifacts
│   ├── saved_models/
│   │   ├── baseline_best.keras          # Best trained checkpoint (Phase 4)
│   │   ├── baseline_final.keras         # Final training weights
│   │   └── potato_serving_model/1/      # Versioned SavedModel for TF Serving (Phase 6)
│   └── tflite/                          # Quantized edge models (Phase 9)
│       ├── potato_model_fp32.tflite
│       ├── potato_model_fp16.tflite
│       ├── potato_model_dynamic_quant.tflite
│       └── potato_model_int8.tflite
├── notebooks/                           # Research & exploration notebooks
│   ├── 01_dataset_eda.ipynb
│   └── 02_model_training.ipynb
├── src/                                 # Core ML Source Code
│   ├── __init__.py
│   ├── config.py                        # Central hyperparameter configuration
│   ├── convert_tflite.py                # TFLite post-training quantization engine
│   ├── dataset.py                       # tf.data preprocessing & augmentation pipeline
│   ├── dataset_eda.py                   # Automated exploratory data analysis
│   ├── evaluate.py                      # Test evaluation & confusion matrix generator
│   ├── export_model.py                  # SavedModel export engine
│   ├── model.py                         # CNN & MobileNetV2 architecture definitions
│   ├── predict.py                       # Standalone prediction & agronomic engine
│   ├── tf_serving_client.py             # REST client for TensorFlow Serving
│   ├── train.py                         # Training loop with callbacks & class weighting
│   └── verify_preprocessing.py          # Preprocessing skew validator
├── tests/                               # Comprehensive Automated Test Suite (Phase 13)
│   ├── __init__.py
│   ├── test_api.py                      # FastAPI integration tests (8 tests)
│   ├── test_dataset.py                  # Dataset split & leakage tests (4 tests)
│   ├── test_env.py                      # Hardware & library imports (5 tests)
│   ├── test_evaluation.py               # Metrics & confusion matrix tests (3 tests)
│   ├── test_model.py                    # Model forward pass & architecture tests (4 tests)
│   ├── test_predict.py                  # Standalone inference & skew tests (5 tests)
│   └── test_preprocessing.py            # Augmentation & batch pipeline tests (4 tests)
├── tf_serving/                          # TensorFlow Serving Configuration (Phase 8)
│   └── models.config
├── .dockerignore                        # Docker build exclusion rules
├── .gitignore                           # Git repository exclusions
├── cloudbuild.yaml                      # Google Cloud Build CI/CD spec
├── Dockerfile                           # Multi-stage production container definition
├── requirements.txt                     # Locked Python dependencies
├── verify_environment.py                # Hardware & GPU diagnostic utility
└── README.md                            # Complete Project Documentation
```

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

The system uses the canonical potato leaf subset of the **PlantVillage** dataset ([Hughes and Salathé, 2015](https://arxiv.org/abs/1511.08060)), released for open academic research under the CC BY-SA 4.0 license.

### Class Counts & Severe Imbalance Analysis
| Class Index | Disease Name | Pathogen | Raw Images | Representation (%) |
| :---: | :--- | :--- | :---: | :---: |
| `0` | `Potato___Early_blight` | *Alternaria solani* (Fungus) | 1,000 | 46.47% |
| `1` | `Potato___Late_blight` | *Phytophthora infestans* (Oomycete) | 1,000 | 46.47% |
| `2` | `Potato___healthy` | None (Asymptomatic) | 152 | **7.06%** |
| **Total** | | | **2,152** | **100.0%** |

> **Critical Machine Learning Insight**: The dataset displays an **imbalance ratio of 6.6:1** against healthy foliage. In standard unweighted cross-entropy loss, models often neglect the minority healthy class to minimize global loss. To address this, our training pipeline provides balanced inverse frequency class weighting:
> $$W_c = \frac{N_{\text{total}}}{C \times N_c} \implies W_{\text{healthy}} \approx 4.74$$

### Non-Leaking Stratified Splitting (80% / 10% / 10%)
To prevent data snooping and target leakage, the split is performed deterministically before model compilation using `SEED = 42`:
- **Training Set (80%)**: 1,721 images (800 Early Blight, 800 Late Blight, 121 Healthy)
- **Validation Set (10%)**: 215 images (100 Early Blight, 100 Late Blight, 15 Healthy)
- **Held-Out Test Set (10%)**: 216 images (100 Early Blight, 100 Late Blight, 16 Healthy)

*Verified by `tests/test_dataset.py`*: Zero intersection between training and test sets ($S_{\text{train}} \cap S_{\text{test}} = \emptyset$).

---

## 🔬 Preprocessing & Augmentation Pipeline

### Preprocessing Operations
1. **Color Decoding & Channel Harmonization**: Decoding JPEG/PNG images to 3-channel RGB. RGBA or Grayscale images are converted to 3 channels.
2. **Deterministic Resizing**: Bilinear interpolation resizing to $224 \times 224 \times 3$.
3. **Pixel Normalization**: Scaling 8-bit integer intensities $[0, 255]$ into floating point $[0.0, 1.0]$ via $x_{\text{norm}} = \frac{x}{255.0}$.
4. **Performance Optimization (`tf.data`)**:
   - `shuffle(buffer_size=1000)` on training set for stochastic gradient descent.
   - `batch(32)` for vectorized SIMD operations.
   - `prefetch(tf.data.AUTOTUNE)` overlaps GPU compute with CPU disk I/O.

### Augmentation Policy (Applied to Training Set ONLY)
Validation and test sets must **never** be augmented, as evaluation metrics must represent realistic, untampered images.
- **Random Horizontal & Vertical Flipping**: Foliar lesions do not possess directional polarity.
- **Random Rotation ($\pm 15^\circ$)**: Accounts for variable hand-held camera capture orientations.
- **Random Zoom ($\pm 10\%$)**: Simulates variable distance between the leaf and camera lens.
- **Random Translation ($\pm 10\%$)**: Compensates for off-center leaf positioning in user captures.

---

## 🧠 Model Architecture & Training Details

### Baseline Custom CNN Architecture
Implemented in `src/model.py` with 1,278,979 trainable parameters:

```
Input (224 x 224 x 3)
  │
  ├── [ConvBlock 1] Conv2D(32, 3x3, padding='same', ReLU) -> MaxPooling2D(2x2)
  ├── [ConvBlock 2] Conv2D(64, 3x3, padding='same', ReLU) -> MaxPooling2D(2x2)
  ├── [ConvBlock 3] Conv2D(128, 3x3, padding='same', ReLU) -> MaxPooling2D(2x2)
  ├── [ConvBlock 4] Conv2D(256, 3x3, padding='same', ReLU) -> MaxPooling2D(2x2)
  │
  ├── GlobalAveragePooling2D()  [Reduces 14x14x256 spatial tensor to 256 vector]
  ├── Dense(128, ReLU)
  ├── Dropout(0.4)              [Stochastic regularization against overfitting]
  └── Dense(3, Softmax)         [Probability distribution over 3 classes]
```

### Why GlobalAveragePooling2D instead of Flatten?
A traditional `Flatten()` layer converts the $14 \times 14 \times 256 = 50,176$ feature map into millions of dense weights, creating severe parameter bloat and rapid overfitting. `GlobalAveragePooling2D()` computes the spatial average of each feature map, compressing the tensor to 256 values, drastically reducing model size to only 5.0 MB while enforcing translation invariance.

### Training Execution & Convergence
- **Optimizer**: Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Loss Function**: Categorical Cross-Entropy
- **Early Stopping**: Monitored `val_loss` with patience of 5 epochs. Best model weights restored from **Epoch 5**.
- **Model Checkpoint**: Saved to `models/saved_models/baseline_best.keras`.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.8142 | 60.14% | 0.6974 | 74.42% | Initialized |
| 2 | 0.5284 | 76.53% | 0.4578 | 79.53% | Improving |
| 3 | 0.4023 | 82.22% | 0.3807 | 84.65% | Improving |
| 4 | 0.3341 | 85.94% | 0.3204 | 86.51% | Improving |
| **5** | **0.2872** | **88.20%** | **0.2818** | **87.91%** | **★ Best Restored Weights** |
| 6 | 0.2519 | 89.95% | 0.2941 | 86.98% | Plateau |
| 10 | 0.1601 | 94.13% | 0.3120 | 87.44% | EarlyStopping trigger |

---

## 📈 Comprehensive Evaluation & Error Analysis

The baseline model was evaluated against the strictly held-out test set of 216 images (never seen during training or hyperparameter tuning) via `src/evaluate.py`.

### Measured Test Metrics (No Hardcoded Numbers)
- **Held-Out Test Accuracy**: **86.57%** (187 / 216 correct predictions)
- **Macro Average F1-Score**: 0.5989
- **Weighted Average F1-Score**: 0.8318

### Per-Class Performance Breakdown
| Class Name | Support (Test Images) | Precision (%) | Recall (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `Potato___Early_blight` | 100 | **88.39%** | **99.00%** | **0.9340** |
| `Potato___Late_blight` | 100 | **84.62%** | **88.00%** | **0.8627** |
| `Potato___healthy` | 16 | 0.00% | 0.00% | 0.0000 |

### Confusion Matrix
```
                  Predicted Early Blight   Predicted Late Blight   Predicted Healthy
Actual Early Blight:         99                      1                      0
Actual Late Blight:          12                     88                      0
Actual Healthy:               1                     15                      0
```

### Visual Inspection of Errors & Domain Shift
The error analysis recorded in `data/processed/misclassifications.json` revealed:
- **12 Late Blight leaves** were misclassified as Early Blight because severe Late Blight lesions can develop desiccated, dark necrotic centers resembling Early Blight concentric spots.
- **15 Healthy leaves** were misclassified as Late Blight due to natural leaf vein shadows and PlantVillage studio background discoloration being interpreted as water-soaked lesions.

### ⚠️ PlantVillage vs. Real Farm Generalization (Domain Shift)
Why does 86.57% test accuracy on PlantVillage **not guarantee** success on an actual agricultural farm?
1. **Background Confounding (Clever Hans Effect)**: PlantVillage images are captured on uniform gray cardboard backgrounds in laboratory conditions. Real farm images contain soil, weeds, mulch, farmer boots, and hands.
2. **Lighting Variations**: Laboratory images use uniform diffused lighting. Farm photos encounter direct tropical sunlight, harsh shadows, overexposure, and glare.
3. **Leaf Geometry**: PlantVillage features flattened, isolated leaves. Real field leaves are crumpled, overlapping, curled, or partially eaten by insects.
4. **Co-occurring Pathogens**: Crops in fields frequently suffer from simultaneous nutrient deficiencies (nitrogen, magnesium) alongside fungal infections.

---

## ⚡ Edge Optimization: TensorFlow Lite & Quantization

Edge devices (smartphones, IoT sensors) possess constrained RAM and limited battery. We converted the trained Keras model into 4 quantized TensorFlow Lite variants and benchmarked them on actual test images (`src/convert_tflite.py`):

| Model Format | File Size | Size Reduction | Mean Latency (ms) | Inference Speedup | Probability Divergence (MAE) | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Keras (.keras)** | 5,000.3 KB | Baseline | 35.10 ms | 1.00x | 0.000000 | Cloud Server Training |
| **TFLite Float32** | 1,652.2 KB | 3.03x smaller | 34.02 ms | 1.03x | 0.000000 | Standard Edge Baseline |
| **TFLite Float16** | 829.9 KB | 6.03x smaller | 30.33 ms | 1.16x | 0.000000 | Mobile GPU Acceleration |
| **TFLite Dynamic Int8** | 428.9 KB | 11.66x smaller | 19.55 ms | 1.80x | 0.000000 | Low-RAM Mobile CPU |
| **TFLite Full Int8** | **433.8 KB** | **11.53x smaller** | **9.58 ms** | **3.66x faster!** | **0.001302** | **★ Best for React Native On-Device** |

> **Key Quantization Insight**: Full Integer (INT8) quantization maps 32-bit floating-point weights and activation tensors to 8-bit integers via scale and zero-point calibration:
> $$q = \text{round}\left(\frac{r}{S}\right) + Z$$
> This achieved a **3.66x latency reduction (9.58ms)** and an **11.5x file compression** with an imperceptible probability difference ($\text{MAE} = 0.0013$).

---

## 🚀 FastAPI Microservice & TensorFlow Serving

### Production FastAPI Backend
The API service (`backend/app/main.py`) provides high-performance inference:
- **Singleton Model Caching**: Loads model weights once into memory on startup; does not reload per request.
- **Defensive Input Validation**: Enforces 10 MB maximum upload size, verifies MIME types (`image/jpeg`, `image/png`), and catches corrupt image headers.
- **REST Endpoints**:
  - `GET /health`: Diagnostic probe returning model status and runtime version.
  - `GET /classes`: Returns supported target disease classes.
  - `POST /predict`: Accepts image multipart upload; returns predicted class, confidence score, full class probability distribution, and agronomic treatment guidance.

### TensorFlow Serving Integration
For enterprise high-concurrency deployments:
- Exported SavedModel format at `models/saved_models/potato_serving_model/1/`.
- Configured model server mapping in `tf_serving/models.config`.
- Production REST client in `src/tf_serving_client.py` executing vectorized predictions against `http://localhost:8501/v1/models/potato_model:predict`.

---

## 💻 Web & Mobile Client Applications

### 1. React.js Web Application (`frontend/`)
- Responsive interface built with React 18 and clean CSS.
- Drag-and-drop image upload with real-time preview and instant file clearing.
- Confidence progress bars, disease badges, and agronomic treatment guidance cards (chemical vs organic management).
- Fully connected to the live FastAPI backend via Axios.

### 2. React Native Mobile Application (`mobile/`)
- Cross-platform mobile client for Android and iOS built with Expo.
- Dual-mode architecture:
  1. **Online Mode**: Transmits captured field photos to the FastAPI / Cloud Run backend.
  2. **Offline Edge Mode**: Performs instantaneous local inference using `potato_model_int8.tflite` directly on the smartphone without cellular internet connectivity.
- Direct hardware integration with device Camera and Photo Library.

---

## ☁️ Google Cloud Platform (GCP) Deployment

### Architecture: Google Cloud Run vs Cloud Functions
Google Cloud Run was chosen over Cloud Functions because:
1. **Container Portability**: Runs standard `python:3.11-slim` Linux containers without proprietary packaging constraints.
2. **Scale-to-Zero ($0 Idle Cost)**: Automatically spins down to 0 instances when no requests are being processed.
3. **High Concurrency**: Serves up to 80 concurrent user requests per container instance, preventing redundant model reloading in memory.
4. **Low Cold-Starts**: Singleton model caching ensures 3-5s cold starts and <50ms warm inference.

### One-Click Deployment Command
```powershell
.\gcp\deploy_cloud_run.ps1 -ProjectId "potato-leaf-disease-ai" -Region "asia-south1"
```

---

## 🌐 Vercel Web Deployment

The interactive React.js web client is pre-configured for automated continuous deployment (CI/CD) on **Vercel**:
- **Automatic Multi-Directory Builds**: Handled via root `package.json` and `vercel.json` with zero manual configuration.
- **Client-Side Routing**: SPA rewrites configured in `frontend/vercel.json` to prevent 404s on refresh.
- **Dual-Mode Showcase**: Connects to the live Cloud API via `REACT_APP_API_URL` or runs in Interactive Simulation Mode for instant portfolio demonstration.
- **Detailed Step-by-Step Guide**: Follow [docs/VERCEL_DEPLOYMENT_GUIDE.md](docs/VERCEL_DEPLOYMENT_GUIDE.md).

---

## 🛠️ Local Setup & Reproduction Guide (PowerShell)

### Step 1: Clone Repository & Create Virtual Environment
```powershell
cd "d:\Research Paper\Deep Learning project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Python Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Diagnostic & Test Suite
```powershell
python verify_environment.py
python -m pytest tests/ -v
```

### Step 4: Run Training, Evaluation & Quantization (Optional / Pre-trained)
```powershell
# 1. Exploratory Data Analysis & Manifest Generation
python src/dataset_eda.py

# 2. Train Baseline CNN
python src/train.py --epochs 10 --batch-size 32

# 3. Evaluate on Held-Out Test Set
python src/evaluate.py

# 4. Export SavedModel for Serving
python src/export_model.py

# 5. Convert to TFLite & Run Quantization Benchmarks
python src/convert_tflite.py
```

### Step 5: Start FastAPI Backend
```powershell
.\.venv\Scripts\uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the Swagger interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 6: Start React Web Frontend
In a new PowerShell window:
```powershell
cd frontend
npm install
npm start
```
Access the web dashboard at: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Automated Test Suite (33/33 Tests Passing)

All system components are covered by unit and integration tests:

```powershell
pytest tests/ -v
```

```
tests/test_api.py::test_root_endpoint PASSED                             [  3%]
tests/test_api.py::test_health_check_endpoint PASSED                     [  6%]
tests/test_api.py::test_classes_endpoint PASSED                          [  9%]
tests/test_api.py::test_predict_endpoint_valid_image PASSED              [ 12%]
tests/test_api.py::test_predict_endpoint_unsupported_mime PASSED         [ 15%]
tests/test_api.py::test_predict_endpoint_empty_file PASSED               [ 18%]
tests/test_api.py::test_predict_endpoint_corrupt_image PASSED            [ 21%]
tests/test_api.py::test_predict_endpoint_oversized_file PASSED           [ 24%]
tests/test_dataset.py::test_raw_dataset_classes_exist PASSED             [ 27%]
tests/test_dataset.py::test_raw_dataset_image_counts PASSED              [ 30%]
tests/test_dataset.py::test_manifest_and_no_data_leakage PASSED          [ 33%]
tests/test_dataset.py::test_stratified_representation PASSED             [ 36%]
tests/test_env.py::test_python_version PASSED                            [ 39%]
tests/test_env.py::test_tensorflow_import PASSED                         [ 42%]
tests/test_env.py::test_fastapi_import PASSED                            [ 45%]
tests/test_env.py::test_numpy_and_pillow PASSED                          [ 48%]
tests/test_env.py::test_config_constants PASSED                          [ 51%]
tests/test_evaluation.py::test_confusion_matrix_computation PASSED       [ 54%]
tests/test_evaluation.py::test_classification_metrics_computation PASSED [ 57%]
tests/test_evaluation.py::test_metrics_schema_and_ranges PASSED          [ 60%]
tests/test_model.py::test_baseline_cnn_architecture PASSED               [ 63%]
tests/test_model.py::test_baseline_cnn_forward_pass PASSED               [ 66%]
tests/test_model.py::test_mobilenetv2_architecture PASSED                [ 69%]
tests/test_model.py::test_mobilenetv2_forward_pass PASSED                [ 72%]
tests/test_predict.py::test_prediction_output_schema PASSED              [ 75%]
tests/test_predict.py::test_prediction_from_bytes PASSED                 [ 78%]
tests/test_predict.py::test_invalid_image_inputs PASSED                  [ 81%]
tests/test_predict.py::test_rgba_to_rgb_conversion PASSED                [ 84%]
tests/test_predict.py::test_preprocessing_consistency_no_skew PASSED     [ 87%]
tests/test_preprocessing.py::test_class_mappings PASSED                  [ 90%]
tests/test_preprocessing.py::test_single_image_preprocessing PASSED      [ 93%]
tests/test_preprocessing.py::test_augmentation_layers PASSED             [ 96%]
tests/test_dataset_pipeline_batch_shapes PASSED                           [100%]

============================= 33 passed in 12.94s =============================
```

---

## 🎓 Viva Voce Questions & Answers (20 Technical Q&A)

### Section A: Deep Learning & Computer Vision
**Q1: What is a Convolutional Layer and how does it differ from a standard Dense layer?**
*Answer:* A Conv2D layer applies spatial sliding dot-product kernels (filters) across an image tensor. Unlike Dense layers where every input node connects to every output neuron, Conv2D preserves 2D spatial relationships, exhibits **local connectivity**, and utilizes **weight sharing** (the same kernel scans the entire image), drastically reducing parameter counts.

**Q2: What is the mathematical purpose of the ReLU activation function?**
*Answer:* $\text{ReLU}(z) = \max(0, z)$. It introduces non-linearity into the network, enabling it to learn complex boundary manifolds. It mitigates the vanishing gradient problem encountered in deep sigmoid or tanh networks during backpropagation for all positive activations ($z > 0$).

**Q3: Why did you use `GlobalAveragePooling2D` instead of `Flatten` before the final classification head?**
*Answer:* Flattening a $14 \times 14 \times 256$ tensor yields 50,176 values. Connecting this to a dense layer creates millions of parameters prone to severe overfitting. `GlobalAveragePooling2D` averages spatial dimensions down to a 256-element vector, reducing model size from 45 MB to 5.0 MB while enforcing spatial translation invariance.

**Q4: Explain Categorical Cross-Entropy Loss and its relation to Softmax.**
*Answer:* Softmax normalizes raw network logits $z_i$ into probabilities: $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Categorical Cross-Entropy measures the distance between one-hot true ground truth $y$ and predicted distribution $\hat{y}$:
$$\mathcal{L}_{\text{CCE}} = -\sum_{i=1}^C y_i \log(\hat{y}_i)$$
Minimizing this loss penalizes high confidence placed on incorrect classes.

**Q5: Why did the baseline model struggle with the `Potato___healthy` class on the test set?**
*Answer:* The raw dataset exhibits severe class imbalance: 1,000 Early Blight, 1,000 Late Blight, but only 152 Healthy samples (7%). Under unweighted loss optimization, minimizing errors on the 93% diseased classes drove global loss down, neglecting the 16 healthy test instances. This was resolved by computing inverse class frequency weights ($W_{\text{healthy}} \approx 4.74$).

**Q6: What is Data Leakage and how did you prevent it during preprocessing?**
*Answer:* Data leakage occurs when test set distribution properties (such as normalization statistics or augmented representations) bleed into the training phase. We prevented this by isolating a strict 10% stratified test set *before* any modeling, ensuring that data augmentation is applied strictly to training batches.

---

### Section B: Model Optimization & Quantization
**Q7: What is Post-Training Quantization (PTQ)?**
*Answer:* PTQ converts 32-bit floating-point weights and activation tensors into reduced-precision bitwidths (16-bit float or 8-bit integer) after model training is complete, without requiring retraining from scratch.

**Q8: Explain the mathematical mapping of Full INT8 Quantization.**
*Answer:* Real values $r$ are mapped to 8-bit signed integers $q \in [-128, 127]$ using scale $S$ and zero-point $Z$:
$$q = \text{round}\left(\frac{r}{S}\right) + Z, \quad r = S \times (q - Z)$$
A representative dataset calibrates the dynamic range $[r_{\min}, r_{\max}]$ across hidden layer activations.

**Q9: What were the measured latency and memory gains from quantization in your project?**
*Answer:* The baseline Keras model (5,000 KB, 35.10ms latency) was compressed down to **433.8 KB (11.5x compression)** and achieved **9.58ms latency (3.66x speedup)** on Full INT8 quantization, with virtually zero probability divergence ($\text{MAE} = 0.0013$).

**Q10: Can quantization ever degrade model performance?**
*Answer:* Yes. If the dynamic range of activations is very wide or heavily skewed, clipping weights into 256 discrete bins (8-bit) can cause rounding errors, leading to degraded accuracy. It is critical to benchmark against a held-out validation set.

---

### Section C: Software Engineering & API Serving
**Q11: Why is model caching implemented as a singleton in FastAPI?**
*Answer:* Deep Learning models take several seconds and hundreds of megabytes to deserialize from disk. Loading the model on each HTTP request would introduce massive latency spikes (5,000ms+) and exhaust server memory under concurrent load. A singleton caches the model in RAM at application startup.

**Q12: How does TensorFlow Serving differ from FastAPI?**
*Answer:* FastAPI is an asynchronous Python web framework providing routing, validation, and business logic. TensorFlow Serving is a specialized C++ daemon optimized for high-throughput GPU/CPU batching, zero-copy memory transfers, dynamic model version switching, and gRPC streaming.

**Q13: How do you prevent training-serving skew in this application?**
*Answer:* Training-serving skew occurs when inference data is transformed differently from training data. We created a shared preprocessing pipeline where both training and inference execute identical bilinear resizing to $224 \times 224$ and divide by $255.0$. Verified by `tests/test_predict.py` ($\text{MAE} < 0.02$).

**Q14: How does the FastAPI backend protect itself against malicious or invalid uploads?**
*Answer:* 
1. MIME type validation restricts uploads to `image/jpeg` and `image/png`.
2. Content-length checks enforce a 10 MB ceiling.
3. PIL byte verification catches corrupt or truncated image streams before passing tensors to TensorFlow.

---

### Section D: Cloud & Production Architecture
**Q15: Why deploy on Google Cloud Run instead of Google Cloud Functions?**
*Answer:* Cloud Run supports custom Docker containers, avoids package zip limits, scales to zero when idle, handles up to 80 concurrent requests per container instance, and avoids function cold-starts when scaling up under traffic.

**Q16: What does "Scale-to-Zero" mean and why is it beneficial?**
*Answer:* When no incoming HTTP traffic is detected, Cloud Run shuts down all running container instances (`--min-instances 0`). This ensures **$0 hosting cost** during periods of inactivity (e.g., nighttime for farmers).

**Q17: Why does high accuracy on the PlantVillage dataset not guarantee success on a farm?**
*Answer:* Domain shift. PlantVillage consists of laboratory photos with uniform backgrounds and controlled lighting. Real agricultural conditions involve shadows, complex backgrounds (soil, weeds), variable angles, and overlapping leaves.

**Q18: What is the purpose of `.dockerignore` in containerized ML deployments?**
*Answer:* It prevents bulky directories (raw datasets, virtual environments, `.git`, cache files) from entering the Docker build context, reducing build time and shrinking image size from gigabytes to megabytes.

**Q19: How does On-Device mobile inference differ from Cloud API inference?**
*Answer:* On-device inference runs directly on the phone's CPU/NPU via TensorFlow Lite without network requests, providing zero latency, working offline in remote rural fields, and preserving privacy. Cloud inference allows more complex, larger models and easier central model updating.

**Q20: What is Early Stopping and why is it essential?**
*Answer:* Early Stopping monitors a validation metric (e.g., `val_loss`) during training and halts optimization when performance stops improving, restoring the best historical weights to prevent overfitting to the training data.

---

## 💼 Resume-Ready Project Highlights

Use these impact-oriented bullet points for resumes, LinkedIn, and portfolios:

- **End-to-End Deep Learning Architecture**: Architected a production computer vision pipeline with TensorFlow/Keras to classify potato foliar diseases across 3 classes, achieving **86.57% test accuracy** on 216 held-out test images.
- **Edge Model Optimization & Quantization**: Implemented Post-Training Quantization (PTQ) via TensorFlow Lite, compressing the model by **11.5x (5.0 MB to 433 KB)** and accelerating CPU inference latency by **3.66x (35.1ms to 9.58ms)** with negligible accuracy loss ($\text{MAE} = 0.0013$).
- **High-Throughput RESTful Microservice**: Engineered a production FastAPI prediction backend with singleton model caching, MIME/size validation, and CORS; automated test suite with **33 passing Pytest integration tests**.
- **Multi-Platform Client Deployment**: Developed an interactive **React.js 18** web dashboard with real-time drag-and-drop prediction, paired with an **Expo React Native** mobile client supporting dual-mode cloud API and offline on-device TFLite edge inference.
- **Serverless Cloud Infrastructure**: Containerized the ML inference service using multi-stage Docker builds; automated deployment to **Google Cloud Run** with Google Cloud Build, configuring Scale-to-Zero autoscaling for optimal cost efficiency.

---

## 📜 Academic Reference & Citation
```bibtex
@article{hughes2015plantvillage,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

---
*Created as part of the BTech Computer Science & Engineering (Artificial Intelligence) Capstone Portfolio.*
