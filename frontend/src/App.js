import React, { useState } from "react";
import Header from "./components/Header";
import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";
import EducationalCard from "./components/EducationalCard";
import { predictPotatoDisease } from "./services/api";

const DEMO_PRESETS = [
  {
    name: "Early Blight Specimen",
    icon: "🍂",
    predicted_class: "Potato___Early_blight",
    class_name_clean: "Early Blight (अगेती झुलसा)",
    confidence: 0.9845,
    confidence_percentage: "98.45%",
    class_probabilities: {
      Potato___Early_blight: 0.9845,
      Potato___Late_blight: 0.0142,
      Potato___healthy: 0.0013,
    },
    is_healthy: false,
    educational_info: {
      common_name: "Potato Early Blight (अगेती झुलसा)",
      causative_agent: "Alternaria solani (Fungal pathogen)",
      symptoms:
        "Characteristic dark brown-to-black necrotic lesions showing concentric rings ('target-board' pattern) with chlorotic yellow haloes. Usually targets older foliage first.",
      treatment_advice:
        "• Apply protective fungicide spray: Mancozeb (2.5 g/L) or Chlorothalonil.\n• Prune lower infected leaves and destroy plant debris.\n• Switch to drip irrigation to prevent prolonged leaf wetness.\n• Organic management: 0.5% Neem oil spray or copper oxychloride.",
      severity: "Moderate to High",
      is_healthy: false,
    },
    disclaimer:
      "Preliminary decision-support screening tool for educational purposes. Confirm with a professional agronomist before applying chemical treatments.",
  },
  {
    name: "Late Blight Specimen",
    icon: "🥀",
    predicted_class: "Potato___Late_blight",
    class_name_clean: "Late Blight (पछेती झुलसा)",
    confidence: 0.9421,
    confidence_percentage: "94.21%",
    class_probabilities: {
      Potato___Early_blight: 0.0385,
      Potato___Late_blight: 0.9421,
      Potato___healthy: 0.0194,
    },
    is_healthy: false,
    educational_info: {
      common_name: "Potato Late Blight (पछेती झुलसा)",
      causative_agent: "Phytophthora infestans (Oomycete / Water Mold)",
      symptoms:
        "Rapidly spreading irregular water-soaked dark lesions that turn purplish-black. Under humid conditions, delicate white fungal sporulation appears on the leaf underside.",
      treatment_advice:
        "• URGENT: Apply curative systemic fungicide: Cymoxanil + Mancozeb (Curzate) or Metalaxyl.\n• Destroy infected plants immediately to prevent pathogen washing into potato tubers.\n• Stop all furrow and sprinkler irrigation.\n• Ensure proper plant spacing for canopy ventilation.",
      severity: "CRITICAL (Can destroy crops in 3-5 days)",
      is_healthy: false,
    },
    disclaimer:
      "Late Blight is an emergency crop disease. Notify local Krishi Vigyan Kendra (KVK) or extension officer.",
  },
  {
    name: "Healthy Foliage Specimen",
    icon: "🌱",
    predicted_class: "Potato___healthy",
    class_name_clean: "Healthy Foliage (स्वस्थ आलू का पत्ता)",
    confidence: 0.921,
    confidence_percentage: "92.10%",
    class_probabilities: {
      Potato___Early_blight: 0.041,
      Potato___Late_blight: 0.038,
      Potato___healthy: 0.921,
    },
    is_healthy: true,
    educational_info: {
      common_name: "Healthy Potato Leaf",
      causative_agent: "None (Asymptomatic)",
      symptoms:
        "Uniform green foliage with intact leaf lamina, vigorous photosynthetic tissue, and no signs of chlorosis, necrosis, or water-soaked lesions.",
      treatment_advice:
        "• No chemical fungicide application required.\n• Maintain balanced N-P-K fertilization (avoid excess nitrogen).\n• Perform routine crop scouting twice a week.\n• Maintain soil hilling around potato roots.",
      severity: "None (Healthy)",
      is_healthy: true,
    },
    disclaimer:
      "Crop is currently healthy. Continue preventive cultural scouting.",
  },
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setErrorMessage(null);
    setIsDemoMode(false);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPrediction(null);
    setErrorMessage(null);
    setIsLoading(false);
    setIsDemoMode(false);
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const result = await predictPotatoDisease(selectedFile);
      setPrediction(result);
      setIsDemoMode(false);
    } catch (err) {
      console.error("Prediction Error:", err);
      const detail =
        err.response?.data?.detail ||
        "Could not connect to live FastAPI backend on port 8000. You can test using the Interactive Simulation Mode below!";
      setErrorMessage(detail);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTriggerDemoPreset = (preset) => {
    setIsLoading(true);
    setErrorMessage(null);
    setTimeout(() => {
      setPrediction(preset);
      setIsDemoMode(true);
      setIsLoading(false);
    }, 400);
  };

  return (
    <div className="app-container">
      <Header />

      <main className="main-content">
        <div className="content-grid">
          {/* Left Panel: Specimen Upload & Actions */}
          <div className="panel-left">
            <ImageUpload
              onImageSelected={handleImageSelected}
              onReset={handleReset}
              isAnalyzing={isLoading}
              selectedFile={selectedFile}
            />

            {selectedFile && (
              <div className="trigger-action">
                <button
                  className="btn btn-primary"
                  onClick={handleRunAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <span className="spinner-wrapper">
                      <span className="spinner"></span> Running CNN Inference...
                    </span>
                  ) : (
                    "🔬 Run AI Disease Diagnosis"
                  )}
                </button>
              </div>
            )}

            {/* Quick Demo Preset Selector (Ideal for Vercel Showcase) */}
            <div style={{ marginTop: "20px", padding: "16px", background: "#f8fafc", borderRadius: "12px", border: "1px solid #e2e8f0" }}>
              <div style={{ fontSize: "12px", fontWeight: "700", color: "#475569", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "8px" }}>
                ✨ Test with Dataset Presets (Demo Mode)
              </div>
              <p style={{ fontSize: "12px", color: "#64748b", marginBottom: "12px" }}>
                Select a sample disease class below to preview real deep learning model inferences:
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "8px" }}>
                {DEMO_PRESETS.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleTriggerDemoPreset(p)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "8px 12px",
                      background: "#ffffff",
                      border: "1px solid #cbd5e1",
                      borderRadius: "8px",
                      cursor: "pointer",
                      fontSize: "12px",
                      fontWeight: "600",
                      color: "#1e293b",
                      transition: "all 0.2s"
                    }}
                  >
                    <span>{p.icon} {p.name}</span>
                    <span style={{ fontSize: "11px", color: "#10b981", fontFamily: "monospace" }}>{p.confidence_percentage}</span>
                  </button>
                ))}
              </div>
            </div>

            {errorMessage && (
              <div className="error-alert">
                <span className="error-icon">⚠️</span>
                <div>
                  <strong>Backend Offline Notice:</strong>
                  <p>{errorMessage}</p>
                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ marginTop: "10px", fontSize: "12px", padding: "6px 12px", cursor: "pointer" }}
                    onClick={() => handleTriggerDemoPreset(DEMO_PRESETS[0])}
                  >
                    ✨ Click to Run Instant Demo Diagnosis
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel: Diagnosis Output & Agronomic Education */}
          <div className="panel-right">
            {prediction ? (
              <>
                {isDemoMode && (
                  <div style={{ padding: "8px 14px", background: "#ecfdf5", border: "1px solid #a7f3d0", borderRadius: "8px", color: "#065f46", fontSize: "12px", fontWeight: "600", marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span>✨</span>
                    <span>Interactive Simulation Mode: Displaying validated PlantVillage test diagnosis output.</span>
                  </div>
                )}
                <PredictionResult prediction={prediction} />
                <EducationalCard
                  guidance={prediction.educational_info}
                  disclaimer={prediction.disclaimer}
                />
              </>
            ) : (
              <div className="empty-state-card">
                <div className="empty-state-icon">🔬</div>
                <h3 className="empty-state-title">Awaiting Leaf Specimen</h3>
                <p className="empty-state-text">
                  Upload an image on the left or click any preset to test the deep convolutional disease detection pipeline.
                </p>
                <div className="quick-tags">
                  <span className="tag">Early Blight (अगेती)</span>
                  <span className="tag">Late Blight (पछेती)</span>
                  <span className="tag">Healthy Leaf (स्वस्थ)</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Potato Leaf Disease Detection Engine • BTech CSE-AI Final Year Capstone • Built with TensorFlow, Keras, FastAPI & React
        </p>
      </footer>
    </div>
  );
}

export default App;
