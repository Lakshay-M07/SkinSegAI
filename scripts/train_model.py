
import argparse
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.lesion_dataset import LesionDataset
from src.models.model_factory import (
    build_model,
    uses_imagenet_normalization,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_ARCHITECTURES,
)
from src.training.model_trainer import ModelTrainer


def main():

    parser = argparse.ArgumentParser(
        description="Train SkinSegAI lesion segmentation model."
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet34_unet",
        choices=SUPPORTED_ARCHITECTURES,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.4,
        help="Weight for Soft Dice loss component (default: 0.4).",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.3,
        help="Weight for BCE with logits loss component (default: 0.3).",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.3,
        help="Weight for Boundary-Aware loss component (default: 0.3).",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if any(w < 0 for w in (args.dice_weight, args.bce_weight, args.boundary_weight)):
        parser.error("Loss weights must be non-negative.")

    if (args.dice_weight + args.bce_weight + args.boundary_weight) == 0:
        parser.error("At least one loss weight must be positive.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]
    splits = root / "data/splits_grouped"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- normalization selection ---
    if uses_imagenet_normalization(args.arch):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = None, None

    train_dataset = LesionDataset(
        csv_path=splits / "train.csv",
        project_root=root,
        augment=True,
        mean=mean,
        std=std,
    )

    val_dataset = LesionDataset(
        csv_path=splits / "validation.csv",
        project_root=root,
        augment=False,
        mean=mean,
        std=std,
    )

    output_dir = root / "artifacts/checkpoints"
    epochs = args.epochs

    if args.smoke:

        train_dataset = Subset(
            train_dataset,
            range(min(8, len(train_dataset)))
        )

        val_dataset = Subset(
            val_dataset,
            range(min(4, len(val_dataset)))
        )

        epochs = 1
        output_dir = output_dir / "smoke_test"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(args.arch)

    trainer = ModelTrainer(
        model=model,
        device=device,
        output_dir=output_dir,
        learning_rate=0.001,
        patience=7,
        arch=args.arch,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        boundary_weight=args.boundary_weight,
    )

    print("\n===== TRAINING CONFIGURATION =====")
    print("Architecture:", args.arch)
    print("Device:", device)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", epochs)
    print("ImageNet normalization:",
          uses_imagenet_normalization(args.arch))
    print("Loss weights:",
          f"Dice={args.dice_weight}, "
          f"BCE={args.bce_weight}, "
          f"Boundary={args.boundary_weight}")
    print("Smoke test:", args.smoke)

    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs
    )


if __name__ == "__main__":
    main()
