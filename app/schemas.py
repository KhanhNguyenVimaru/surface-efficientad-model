from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_path: str


class PredictResponse(BaseModel):
    filename: str
    pred_score: float = Field(..., description="Anomaly score from the model")
    threshold: float
    is_anomaly: bool
    original_image_base64: str
    heatmap_base64: str
    overlay_base64: str
