import React, { useEffect, useState } from "react";
import { checkHealth } from "../services/api";

const Header = () => {
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    const verifyBackend = async () => {
      try {
        const data = await checkHealth();
        if (data.status === "healthy" && data.model_loaded) {
          setBackendStatus("connected");
        } else {
          setBackendStatus("degraded");
        }
      } catch (err) {
        setBackendStatus("disconnected");
      }
    };

    verifyBackend();
    const interval = setInterval(verifyBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="app-header">
      <div className="header-container">
        <div className="header-brand">
          <div className="brand-icon">🥔</div>
          <div>
            <h1 className="brand-title">Potato Disease AI</h1>
            <p className="brand-subtitle">Smart Crop Health & Pathology Screening</p>
          </div>
        </div>

        <div className="status-indicator">
          <span
            className={`status-dot ${
              backendStatus === "connected"
                ? "status-online"
                : backendStatus === "checking"
                ? "status-pending"
                : "status-offline"
            }`}
          ></span>
          <span className="status-text">
            {backendStatus === "connected"
              ? "FastAPI DL Engine Active"
              : backendStatus === "checking"
              ? "Connecting..."
              : "API Offline (127.0.0.1:8000)"}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
