from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from torchvision.transforms import v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientAD on MVTec AD")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to MVTecAD root")
    parser.add_argument("--category", type=str, required=True, help="MVTec category, e.g. bottle")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["s", "m", "small", "medium"],
        help="EfficientAD size. Legacy aliases: s->small, m->medium",
    )
    return parser.parse_args()


def normalize_model_size(raw_value: str) -> str:
    return {"s": "small", "m": "medium"}.get(raw_value, raw_value)


def validate_data_root(data_root: Path, category: str) -> None:
    category_dir = data_root / category
    if not category_dir.exists():
        # Let anomalib auto-download dataset when category folder does not exist.
        return

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_count = sum(
        1
        for file_path in category_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in image_exts
    )
    if image_count == 0:
        raise RuntimeError(
            f"No images found in '{category_dir}'. "
            "Expected MVTecAD structure like <root>/<category>/train/good/*.png and <root>/<category>/test/... "
            "Please pass the real dataset root, not a training results folder."
        )


def main() -> None:
    args = parse_args()
    resize = v2.Resize((args.image_size, args.image_size))
    validate_data_root(args.data_root, args.category)

    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        train_augmentations=resize,
        val_augmentations=resize,
        test_augmentations=resize,
    )

    model = EfficientAd(model_size=normalize_model_size(args.model_size))
    engine = Engine(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=Path("./results") / f"efficientad_{args.category}",
    )

    engine.fit(model=model, datamodule=datamodule)
    predictions = engine.predict(model=model, datamodule=datamodule)
    print(f"Finished training. Predictions batches: {len(predictions)}")
    print("Check the results directory for checkpoints and logs.")


if __name__ == "__main__":
    main()
