import React, { useRef, useState } from "react";

const ImageUpload = ({ onImageSelected, onReset, isAnalyzing, selectedFile }) => {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      validateAndProcess(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      validateAndProcess(file);
    }
  };

  const validateAndProcess = (file) => {
    const validTypes = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
    if (!validTypes.includes(file.type)) {
      alert("Invalid file format. Please upload a JPEG or PNG image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert("File is too large. Maximum size is 10 MB.");
      return;
    }
    onImageSelected(file);
  };

  return (
    <div className="upload-card">
      <h2 className="section-title">Upload Leaf Specimen</h2>
      <p className="section-description">
        Provide a clear, well-lit photo of an individual potato leaf displaying suspected lesion symptoms or healthy foliage.
      </p>

      {!selectedFile ? (
        <div
          className={`dropzone ${dragOver ? "dropzone-active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp"
            style={{ display: "none" }}
          />
          <div className="dropzone-icon">🍃</div>
          <p className="dropzone-text">
            <strong>Click to upload</strong> or drag and drop image here
          </p>
          <span className="dropzone-hint">Supports: JPEG, PNG, WebP (Max: 10MB)</span>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-wrapper">
            <img
              src={URL.createObjectURL(selectedFile)}
              alt="Leaf Specimen Preview"
              className="preview-image"
            />
          </div>
          <div className="preview-meta">
            <span className="file-name">{selectedFile.name}</span>
            <span className="file-size">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </span>
          </div>

          <div className="action-buttons">
            <button
              className="btn btn-secondary"
              onClick={onReset}
              disabled={isAnalyzing}
            >
              🔄 Change Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
