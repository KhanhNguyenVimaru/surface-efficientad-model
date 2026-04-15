from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def read_image_from_bytes(raw: bytes) -> Image.Image:
    image = Image.open(BytesIO(raw)).convert("RGB")
    return image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_val = float(anomaly_map.min())
    max_val = float(anomaly_map.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    normalized = (anomaly_map - min_val) / (max_val - min_val)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def anomaly_map_to_heatmap(anomaly_map: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    normalized = normalize_map(anomaly_map)
    resized = cv2.resize(normalized, output_size, interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def overlay_heatmap_on_image(image: Image.Image, heatmap: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = np.array(image).astype(np.float32)
    overlay = np.array(heatmap).astype(np.float32)
    mixed = (base * (1.0 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mixed)
