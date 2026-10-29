# 🚀 Vercel Deployment Guide — Potato Leaf Disease Detection

This guide walks you through deploying the **Potato Leaf Disease Detection Web Application** on **Vercel** in less than 2 minutes using your GitHub repository.

---

## 🌟 Why Vercel?
- **Instant Global CDN**: Super fast leaf diagnosis UI anywhere in the world.
- **Zero Cost (Free Tier)**: 100% free hosting with automated SSL/TLS certificates (`https://`).
- **Continuous Deployment (CI/CD)**: Every time you run `git push origin main`, Vercel automatically redeploys your updated application.

---

## 📋 Prerequisites
1. A free account on [vercel.com](https://vercel.com).
2. Your project pushed to GitHub: [https://github.com/Ayush196043/Deep-Learning-Project](https://github.com/Ayush196043/Deep-Learning-Project).

---

## 🛠️ Step-by-Step Vercel Deployment (2 Minutes)

### Step 1: Sign in to Vercel
1. Open [https://vercel.com](https://vercel.com) in your web browser.
2. Sign in using your **GitHub account** (`Ayush196043`).

---

### Step 2: Import Your GitHub Repository
1. In your Vercel Dashboard, click **"Add New..."** (top right) and select **"Project"**.
2. Under "Import Git Repository", find **`Deep-Learning-Project`** from your repository list.
3. Click the **"Import"** button next to it.

---

### Step 3: Configure Project Settings (Important!)

On the "Configure Project" screen, configure the following:

1. **Project Name**: `potato-leaf-disease-ai` (or any name you like).
2. **Framework Preset**: Select **`Create React App`**.
3. **Root Directory**:
   - Click **"Edit"** next to **Root Directory**.
   - Select the **`frontend`** directory and click **Continue**.
   *(Note: Because we also added a root `vercel.json`, Vercel will build automatically even if you leave this as default!).*
4. **Build and Output Settings** (Leave as default):
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
5. **Environment Variables** (Optional for Live Cloud Backend):
   - Expand the **Environment Variables** section.
   - **Key**: `REACT_APP_API_URL`
   - **Value**: Your live backend URL (e.g., `https://potato-disease-api-xyz-el.a.run.app` from GCP Cloud Run).
   *(If you don't add this variable yet, the frontend will automatically run in Interactive Simulation Mode so recruiters and evaluators can still test all diseases immediately!).*

---

### Step 4: Click Deploy! 🚀
1. Click the blue **"Deploy"** button.
2. Vercel will install dependencies, build the React production bundle, and deploy it to a global edge network.
3. Within 45–60 seconds, you will see confetti and your live public URL, e.g.:
   `https://potato-leaf-disease-ai.vercel.app`

---

## 🔬 How the Vercel App Works (Dual-Mode Design)

1. **With Cloud Backend Connected**:
   - If `REACT_APP_API_URL` points to your Google Cloud Run API or Render API, leaf uploads are sent directly to the CNN model in the cloud.
2. **Without Cloud Backend (Zero-Config Showcase)**:
   - Evaluators and college faculty can click the **✨ Dataset Presets** on the left to see instant diagnostic outputs, confidence gauges, and treatment recommendations for Early Blight, Late Blight, and Healthy leaves.
