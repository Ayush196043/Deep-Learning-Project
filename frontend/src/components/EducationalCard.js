import React from "react";

const EducationalCard = ({ guidance, disclaimer }) => {
  if (!guidance) return null;

  const {
    common_name,
    causative_agent,
    symptoms,
    treatment_advice,
    severity,
    is_healthy,
  } = guidance;

  return (
    <div className="education-card">
      <div className="card-badge">🌾 Agronomic Advisory & Pathology Guide</div>

      <div className="info-grid">
        <div className="info-block">
          <span className="info-label">Causative Pathogen:</span>
          <p className="info-content">{causative_agent}</p>
        </div>

        <div className="info-block">
          <span className="info-label">Severity Level:</span>
          <p className={`info-content ${is_healthy ? "text-green" : "text-amber"}`}>
            {severity}
          </p>
        </div>
      </div>

      <div className="section-block">
        <h4 className="block-title">🔍 Clinical Diagnostic Symptoms:</h4>
        <p className="block-text">{symptoms}</p>
      </div>

      <div className="section-block">
        <h4 className="block-title">🧪 Recommended Cultural & Fungicide Treatments:</h4>
        <div className="treatment-list">
          {treatment_advice.split("\n").map((step, idx) => (
            <p key={idx} className="treatment-item">
              {step}
            </p>
          ))}
        </div>
      </div>

      {disclaimer && (
        <div className="disclaimer-banner">
          <span className="disclaimer-icon">⚠️</span>
          <p className="disclaimer-text">{disclaimer}</p>
        </div>
      )}
    </div>
  );
};

export default EducationalCard;
