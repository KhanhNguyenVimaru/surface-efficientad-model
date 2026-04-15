from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "EfficientAD Surface Inspection API")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "./checkpoints/model.pt"))
    device: str = os.getenv("DEVICE", "auto")
    default_threshold: float = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    trust_remote_code: str = os.getenv("TRUST_REMOTE_CODE", "1")

    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        return [item.strip() for item in raw.split(",") if item.strip()]


settings = Settings()
