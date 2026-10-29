"""
Pydantic Schemas for API Request and Response Models
Potato Leaf Disease Detection Backend
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: str = Field(..., json_schema_extra={"example": "baseline_best.keras"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})

class ClassesResponse(BaseModel):
    classes: List[str]
    count: int

class EducationalInfo(BaseModel):
    common_name: Optional[str] = None
    causative_agent: Optional[str] = None
    symptoms: Optional[str] = None
    treatment_advice: Optional[str] = None
    severity: Optional[str] = None
    is_healthy: Optional[bool] = None

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., json_schema_extra={"example": "Potato___Early_blight"})
    class_name_clean: str = Field(..., json_schema_extra={"example": "Early Blight (अगेती झुलसा)"})
    confidence: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.9845})
    confidence_percentage: str = Field(..., json_schema_extra={"example": "98.45%"})
    class_probabilities: Dict[str, float]
    is_healthy: bool = Field(..., json_schema_extra={"example": False})
    educational_info: EducationalInfo
    disclaimer: str

class ErrorResponse(BaseModel):
    detail: str
