from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_service import model_service
from app.schemas import HealthResponse, PredictResponse
from app.utils import pil_to_base64, read_image_from_bytes

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        model_service.load()
    except (FileNotFoundError, RuntimeError) as exc:
        # Allow API startup without a model so the user can wire UI first.
        logger.warning("Model was not loaded at startup: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        model_loaded=model_service.is_loaded,
        model_path=str(model_service.model_path),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File(description="Surface image")]) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = read_image_from_bytes(raw)
        result = model_service.predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc

    return PredictResponse(
        filename=file.filename,
        pred_score=result["pred_score"],
        threshold=result["threshold"],
        is_anomaly=result["is_anomaly"],
        original_image_base64=pil_to_base64(image),
        heatmap_base64=pil_to_base64(result["heatmap"]),
        overlay_base64=pil_to_base64(result["overlay"]),
    )
