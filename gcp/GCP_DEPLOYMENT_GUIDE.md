# Google Cloud Platform (GCP) Deployment Guide

This guide provides end-to-end instructions for deploying the **Potato Leaf Disease Detection** production API container to **Google Cloud Run** with artifact storage on **Google Cloud Storage (GCS)**, continuous container builds via **Google Cloud Build**, and observability via **Google Cloud Logging / Monitoring**.

---

## 🏗️ Architecture Overview

```
                      +-----------------------------+
                      |   React Web / Mobile App    |
                      +--------------+--------------+
                                     |
                                     | HTTPS / TLS 1.3 (Port 443)
                                     v
                 +---------------------------------------+
                 |       Google Cloud Run Service        |
                 |   (FastAPI Prediction Microservice)   |
                 |                                       |
                 | - Auto-scaling: 0 to 5 instances      |
                 | - Scale-to-Zero ($0 idle server cost) |
                 | - 2 vCPU, 2 GiB RAM per instance      |
                 | - Request concurrency: 80             |
                 +-------------------+-------------------+
                                     |
                                     | Startup Model Sync / Cache
                                     v
                 +---------------------------------------+
                 |      Google Cloud Storage (GCS)       |
                 |  Bucket: gs://potato-disease-models/  |
                 |  - baseline_best.keras                |
                 |  - potato_model_int8.tflite           |
                 +---------------------------------------+
```

---

## 🎯 Architecture Decision: Why Cloud Run vs Cloud Functions?

| Decision Dimension | Google Cloud Functions (Gen 2) | Google Cloud Run (Chosen Architecture) | Why Cloud Run is Superior for DL |
| :--- | :--- | :--- | :--- |
| **Runtime Containerization** | Limited custom base image layers; zip/tarball execution overhead | Full Docker containerization (`python:3.11-slim`) | Allows bundling optimized C++ BLAS/LAPACK binaries, custom wheels, and locked dependencies without environment drift. |
| **Cold Start Latency** | 8s – 20s when loading heavy TensorFlow / Keras libraries into Python memory | 3s – 6s with pre-warmed container image and single startup singleton model loading | Cloud Run keeps warm containers ready across multiple concurrent requests. |
| **Concurrency & Throughput** | 1 request per function instance (classic FaaS) | Up to 80 concurrent requests per container instance | Severe cost savings. In Cloud Functions, 50 simultaneous user requests spin up 50 separate container runtimes (50x memory usage). Cloud Run handles them within 1-2 instances. |
| **Model Size Headroom** | 512 MB package size limits on standard deployments | Multi-gigabyte containers supported effortlessly | Accommodates large deep neural networks, SavedModel graphs, and dependencies without hitting hard zip quotas. |
| **Local Development Parity** | Local testing requires `functions-framework` emulator | Exact same Docker image runs on local Windows/Linux via `docker run` as it does in GCP | Eliminates "it worked on my laptop" bugs. |

> **Verdict:** Google Cloud Run is the industry standard for production Deep Learning APIs. Cloud Functions is reserved for lightweight event triggers (e.g., resizing images when an agronomist uploads raw pictures to a GCS bucket).

---

## 📋 Prerequisites & Tools

1. **Google Cloud SDK (`gcloud` CLI)**:
   - Install from [cloud.google.com/sdk](https://cloud.google.com/sdk).
2. **Google Cloud Account & Project**:
   - Create a project: e.g., `potato-leaf-disease-ai`.
   - Ensure billing is enabled (Cloud Run has a generous free tier of 2 million requests per month).
3. **Docker Desktop** (optional for local container testing, or use Cloud Build for remote cloud builds).

---

## 🚀 Step-by-Step Production Deployment

### Step 1: Initialize GCP Authentication & Set Active Project

Open Windows PowerShell:
```powershell
# Authenticate with your Google account
gcloud auth login

# Set your active project ID
gcloud config set project potato-leaf-disease-ai

# Verify configuration
gcloud config list
```

### Step 2: Enable Required Google Cloud APIs

Enable the serverless, container registry, and storage services:
```powershell
gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    logging.googleapis.com
```

### Step 3: Create Google Artifact Registry (Docker Repository)

Artifact Registry replaces legacy Container Registry (GCR) with faster, region-localized Docker image repositories:
```powershell
gcloud artifacts repositories create potato-app-repo `
    --repository-format=docker `
    --location=asia-south1 `
    --description="Docker repository for Potato Leaf Disease FastAPI backend"
```

### Step 4: Build Container using Google Cloud Build (Zero Local Docker Required)

Google Cloud Build reads the project's [Dockerfile](file:///d:/Research%20Paper/Deep%20Learning%20project/Dockerfile), packages the application code and model artifacts, and pushes the immutable image to Artifact Registry:
```powershell
# Submit build from project root directory
gcloud builds submit --tag asia-south1-docker.pkg.dev/potato-leaf-disease-ai/potato-app-repo/potato-api:v1.0.0 .
```

*Build highlights:*
- Uses multi-stage caching defined in `Dockerfile`.
- Strips unneeded raw training datasets and virtual environments via `.dockerignore`.
- Bundles [models/saved_models/baseline_best.keras](file:///d:/Research%20Paper/Deep%20Learning%20project/models/saved_models/baseline_best.keras) and [models/tflite/potato_model_int8.tflite](file:///d:/Research%20Paper/Deep%20Learning%20project/models/tflite/potato_model_int8.tflite).

### Step 5: Deploy to Google Cloud Run

Deploy the container with cost-conscious settings (Scale-to-Zero enabled):
```powershell
gcloud run deploy potato-disease-api `
    --image asia-south1-docker.pkg.dev/potato-leaf-disease-ai/potato-app-repo/potato-api:v1.0.0 `
    --platform managed `
    --region asia-south1 `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 5 `
    --concurrency 80 `
    --timeout 60s `
    --set-env-vars="MODEL_PATH=/app/models/saved_models/baseline_best.keras,PORT=8080"
```

#### Parameter Breakdown:
- `--region asia-south1`: Deploys close to Indian farmers and agricultural research centers for low network latency.
- `--min-instances 0`: **Scale-to-Zero**. When there are no user requests at night, instances drop to 0, resulting in **$0 idle server charges**.
- `--max-instances 5`: Hard ceiling prevents runaway costs if the API experiences a traffic surge or DDoS attempt.
- `--memory 2Gi` & `--cpu 2`: Ample headroom for TensorFlow inference and image decoding.
- `--concurrency 80`: Allows a single container instance to serve up to 80 concurrent user requests.
- `--allow-unauthenticated`: Enables public HTTPS access for the React Web and React Native mobile clients.

---

## 🔒 Security, IAM & Governance Best Practices

1. **Principle of Least Privilege**:
   - Create a dedicated runtime Service Account for Cloud Run rather than using the default Compute Engine account:
     ```powershell
     gcloud iam service-accounts create potato-api-runner `
         --description="Service account for Potato Leaf Disease Cloud Run API" `
         --display-name="PotatoApiRunner"
     ```
2. **Read-Only Model Storage Permissions**:
   - If loading models dynamically from GCS at container startup:
     ```powershell
     gsutil mb -p potato-leaf-disease-ai -c standard -l asia-south1 gs://potato-disease-models/
     gsutil cp models/saved_models/baseline_best.keras gs://potato-disease-models/
     gsutil iam ch serviceAccount:potato-api-runner@potato-leaf-disease-ai.iam.gserviceaccount.com:objectViewer gs://potato-disease-models
     ```
3. **CORS Policy Enforcement**:
   - In [backend/app/main.py](file:///d:/Research%20Paper/Deep%20Learning%20project/backend/app/main.py), configure `allow_origins` to restrict access to trusted frontend domains in production instead of `["*"]`.
4. **Non-Root Execution**:
   - Verified in [Dockerfile](file:///d:/Research%20Paper/Deep%20Learning%20project/Dockerfile): Runs as `appuser:appgroup` (UID 10001) to prevent container escape exploits.

---

## 🧪 Testing the Live Cloud Deployment

Once deployed, Google Cloud Run returns a service URL, e.g.:
`https://potato-disease-api-xyz-el.a.run.app`

### 1. Health Probe
```powershell
curl -X GET "https://potato-disease-api-xyz-el.a.run.app/health"
```
*Expected Response:*
```json
{"status":"healthy","model_loaded":true,"model_type":"Keras (.keras)","version":"1.0.0"}
```

### 2. Supported Disease Classes
```powershell
curl -X GET "https://potato-disease-api-xyz-el.a.run.app/classes"
```
*Expected Response:*
```json
{"classes":["Potato___Early_blight","Potato___Late_blight","Potato___healthy"]}
```

### 3. Leaf Disease Prediction
```powershell
curl -X POST "https://potato-disease-api-xyz-el.a.run.app/predict" `
     -H "accept: application/json" `
     -H "Content-Type: multipart/form-data" `
     -F "file=@data/raw/Potato___Early_blight/001000b5-2475-4e29-b422-ad37e78fde5b___RS_Early.B 7051.JPG"
```
*Expected Response:*
```json
{
  "predicted_class": "Potato___Early_blight",
  "confidence": 0.9984,
  "class_probabilities": {
    "Potato___Early_blight": 0.9984,
    "Potato___Late_blight": 0.0015,
    "Potato___healthy": 0.0001
  },
  "explanation": "Early blight caused by Alternaria solani. Target-like dark concentric rings detected.",
  "treatment": {
    "chemical": "Mancozeb or Chlorothalonil fungicide spray",
    "organic": "Neem oil extract, copper fungicides, infected leaf pruning"
  }
}
```

---

## 📊 Monitoring & Log Analytics

View live application logs and inspect response latencies directly via Cloud SDK:
```powershell
# Stream live logs
gcloud beta run services logs tail potato-disease-api --region asia-south1

# Query recent error logs
gcloud logging read "resource.type=cloud_run_revision AND severity>=WARNING" --limit 20
```
