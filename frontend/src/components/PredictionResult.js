import React from "react";

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const {
    predicted_class,
    class_name_clean,
    confidence,
    confidence_percentage,
    class_probabilities,
    is_healthy,
  } = prediction;

  const getStatusTheme = () => {
    if (is_healthy) {
      return {
        badgeClass: "badge-healthy",
        icon: "✅",
        tag: "Healthy Leaf",
      };
    }
    if (predicted_class.includes("Early_blight")) {
      return {
        badgeClass: "badge-early-blight",
        icon: "⚠️",
        tag: "Early Blight Infection",
      };
    }
    return {
      badgeClass: "badge-late-blight",
      icon: "🚨",
      tag: "Late Blight Infection (Critical)",
    };
  };

  const statusTheme = getStatusTheme();

  return (
    <div className="result-card">
      <div className="result-header">
        <span className={`status-badge ${statusTheme.badgeClass}`}>
          {statusTheme.icon} {statusTheme.tag}
        </span>
        <span className="confidence-pill">Confidence: {confidence_percentage}</span>
      </div>

      <h2 className="diagnosis-title">{class_name_clean}</h2>

      <div className="confidence-meter-container">
        <div className="meter-label">
          <span>Prediction Certainty</span>
          <strong>{(confidence * 100).toFixed(1)}%</strong>
        </div>
        <div className="progress-bar-bg">
          <div
            className={`progress-bar-fill ${
              is_healthy ? "fill-green" : predicted_class.includes("Early") ? "fill-orange" : "fill-red"
            }`}
            style={{ width: `${confidence * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="probabilities-section">
        <h3 className="sub-heading">Class Probability Breakdown:</h3>
        <div className="prob-grid">
          {Object.entries(class_probabilities).map(([cls, prob]) => {
            const shortName = cls.replace("Potato___", "").replace("_", " ");
            const pct = (prob * 100).toFixed(1);
            return (
              <div key={cls} className="prob-item">
                <div className="prob-header">
                  <span className="prob-name">{shortName}</span>
                  <span className="prob-value">{pct}%</span>
                </div>
                <div className="prob-track">
                  <div
                    className="prob-fill"
                    style={{ width: `${pct}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
