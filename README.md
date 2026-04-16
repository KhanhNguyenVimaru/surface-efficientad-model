# EfficientAD Training on Google Colab

## 1. Introduction

This project demonstrates how to train the EfficientAD anomaly detection model using the MVTec AD dataset on Google Colab. EfficientAD is a fast anomaly detection method based on a teacher-student architecture combined with an autoencoder.

## 2. Environment Setup

Open Google Colab and create a new notebook. Enable GPU support:

`Runtime -> Change runtime type -> Select GPU`

Verify GPU availability:

```bash
!nvidia-smi
```

## 3. Clone Repository and Install Dependencies

Run the following commands:

```bash
!git clone https://github.com/KhanhNguyenVimaru/surface-efficientad-model.git
%cd surface-efficientad-model
!pip install -r requirements.txt
```

## 4. Dataset Preparation

Download and extract the MVTec AD dataset:

```bash
!wget https://www.mydrive.ch/shares/38536/download/412760263-1629953853/mvtec_anomaly_detection.tar.xz
!tar -xf mvtec_anomaly_detection.tar.xz
```

After extraction, the directory structure should look like:

```text
/content/
  bottle/
  cable/
  capsule/
  ...
```

## 5. Training (Capsule Example)

To train the EfficientAD model on the `capsule` category, run:

```bash
%cd /content/surface-efficientad-model

!python train_efficientad.py \
  --data-root /content \
  --category capsule \
  --max-epochs 30 \
  --batch-size 1 \
  --image-size 256 \
  --devices 1 \
  --accelerator gpu \
  --model-size s
```

## 6. Parameter Description

- `--data-root`: Path to dataset root
- `--category`: Target MVTec category (e.g., `capsule`)
- `--max-epochs`: Number of training epochs
- `--batch-size`: Must be set to `1` for EfficientAD
- `--image-size`: Input image resolution
- `--devices`: Number of GPUs
- `--accelerator`: Hardware type (`gpu` or `cpu`)
- `--model-size`: Model size (`s` or `m`)

## 7. Training Output

After training, results are stored in:

```text
/content/surface-efficientad-model/results/efficientad_capsule/
```

Important files include:

```text
weights/lightning/model.ckpt
images/
```

## 8. Save Model to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!cp /content/surface-efficientad-model/results/efficientad_capsule/EfficientAd/MVTecAD/capsule/v1/weights/lightning/model.ckpt \
/content/drive/MyDrive/capsule.ckpt
```

## 9. Final Output

The final trained model is:

```text
capsule.ckpt
```

This file can be used for inference, deployment, or further evaluation.

## 10. Inference Example

```python
from anomalib.deploy import TorchInferencer

inferencer = TorchInferencer(path="capsule.ckpt")

result = inferencer.predict("test.jpg")

print(result.pred_score)
print(result.pred_label)
```

## 11. Notes

- Each category requires a separate model.
- EfficientAD requires batch size equal to `1`.
- Training is performed using only normal (`good`) images.
- Anomalies are detected based on deviations from learned normal patterns.

## 12. Summary

- Dataset: MVTec AD
- Model: EfficientAD
- Category: Capsule
- Output: `capsule.ckpt`
