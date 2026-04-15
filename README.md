# EfficientAD Surface Inspection Project

This project provides a minimal end-to-end scaffold for:
- training an EfficientAD model with Anomalib
- serving inference through FastAPI
- uploading images from a frontend and visualizing anomaly heatmaps

## Structure

- `backend/app/main.py`: FastAPI app
- `backend/app/model_service.py`: EfficientAD model loader and prediction wrapper
- `backend/app/schemas.py`: API response schemas
- `backend/app/config.py`: environment-driven settings
- `backend/app/utils.py`: image utilities
- `backend/train_efficientad.py`: training script for MVTec AD
- `frontend/index.html`: simple upload UI
- `frontend/app.js`: frontend logic
- `frontend/styles.css`: styling

## 1. Environment

Windows (PowerShell):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train a model

Example with MVTec AD bottle category:

```bash
cd backend
python train_efficientad.py \
  --data-root C:/datasets/MVTecAD \
  --category bottle \
  --max-epochs 20 \
  --model-size small
```

`--data-root` must be the real MVTecAD root containing `<root>/<category>/train/...` and `<root>/<category>/test/...`.
If the category folder does not exist, anomalib will download the dataset to that root.

After training, point `MODEL_PATH` to the exported checkpoint (`.ckpt`) or torch file.

## 3. Run API

```bash
cd backend
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Windows equivalent:

```powershell
cd backend
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Important notes

- For PyTorch model loading through `TorchInferencer`, set `TRUST_REMOTE_CODE=1` because Anomalib warns that torch checkpoint loading uses pickle.
- For stricter production deployment, prefer exporting to OpenVINO and switching the inferencer implementation.
- FastAPI file uploads require `python-multipart`.

## 5. Next steps

- add model export to OpenVINO
- persist upload history
- add threshold calibration endpoint
- replace the demo frontend with React/Vue if needed
