import React, { useState } from "react";
import Header from "./components/Header";
import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";
import EducationalCard from "./components/EducationalCard";
import { predictPotatoDisease } from "./services/api";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setErrorMessage(null);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPrediction(null);
    setErrorMessage(null);
    setIsLoading(false);
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const result = await predictPotatoDisease(selectedFile);
      setPrediction(result);
    } catch (err) {
      console.error("Prediction Error:", err);
      const detail =
        err.response?.data?.detail ||
        "Could not connect to FastAPI server. Please verify backend is running on 127.0.0.1:8000.";
      setErrorMessage(detail);
    } finally {
      setIsLoading(false);
    }
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

            {errorMessage && (
              <div className="error-alert">
                <span className="error-icon">❌</span>
                <div>
                  <strong>Diagnostic Error:</strong>
                  <p>{errorMessage}</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel: Diagnosis Output & Agronomic Education */}
          <div className="panel-right">
            {prediction ? (
              <>
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
                  Upload an image on the left and click <strong>Run AI Disease Diagnosis</strong> to trigger our deep convolutional network model.
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
