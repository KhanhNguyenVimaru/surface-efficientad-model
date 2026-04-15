from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.utils import anomaly_map_to_heatmap, overlay_heatmap_on_image

try:
    from anomalib.deploy import TorchInferencer
except Exception as exc:  # pragma: no cover
    TorchInferencer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(settings.model_path)
        self.device = settings.device
        self.threshold = settings.default_threshold
        self.inferencer: TorchInferencer | None = None

    @property
    def is_loaded(self) -> bool:
        return self.inferencer is not None

    def load(self) -> None:
        if TorchInferencer is None:
            raise RuntimeError(f"Anomalib import failed: {IMPORT_ERROR}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train/export a model first or update MODEL_PATH."
            )
        os.environ["TRUST_REMOTE_CODE"] = settings.trust_remote_code
        self.inferencer = TorchInferencer(path=self.model_path, device=self.device)

    def _ensure_loaded(self) -> TorchInferencer:
        if self.inferencer is None:
            self.load()
        assert self.inferencer is not None
        return self.inferencer

    def predict(self, image: Image.Image) -> dict[str, Any]:
        inferencer = self._ensure_loaded()
        prediction = inferencer.predict(image=image)

        pred_score = float(np.asarray(prediction.pred_score).squeeze())
        anomaly_map = np.asarray(prediction.anomaly_map).squeeze()
        width, height = image.size
        heatmap = anomaly_map_to_heatmap(anomaly_map, (width, height))
        overlay = overlay_heatmap_on_image(image, heatmap)

        return {
            "pred_score": pred_score,
            "threshold": self.threshold,
            "is_anomaly": pred_score >= self.threshold,
            "heatmap": heatmap,
            "overlay": overlay,
        }


model_service = ModelService()
