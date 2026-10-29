# Potato Leaf Disease Detection — React Web Frontend

A modern, responsive React 18 web interface for the Potato Leaf Disease Detection Deep Learning system.

## Features
- **Drag & Drop Specimen Upload:** Instant client-side validation for JPEG/PNG image formats and file size limits (10MB).
- **Real-time Diagnostic Output:** Color-coded disease status badge (Healthy, Early Blight, Late Blight).
- **Certainty Meter & Probability Distribution:** Visual breakdown of softmax confidence scores across all three classes.
- **Agronomic Advisory Cards:** Clinical symptoms, causative fungal pathogens, and actionable cultural/chemical fungicide guidance for farmers.
- **Backend Heartbeat:** Live health indicator tracking FastAPI readiness on `http://127.0.0.1:8000`.

## Quickstart Instructions

### 1. Start the FastAPI Backend First
In your project root terminal:
```powershell
.\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Install Frontend Dependencies & Start React
In a new terminal window inside the `frontend` folder:
```powershell
cd frontend
npm install
npm start
```
The application will launch in your browser at `http://localhost:3000`.
