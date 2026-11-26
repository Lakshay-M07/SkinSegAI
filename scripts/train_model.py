
"""Train SkinSegAI segmentation models."""

import argparse
import math
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.lesion_dataset import LesionDataset

from src.models.artifacts import ModelArtifactRegistry

from src.models.model_factory import (
    build_model,
    uses_imagenet_normalization,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_ARCHITECTURES,
)

from src.training.model_trainer import ModelTrainer


def main():

    # ----------------------------------------------- #
    # Command-line configuration
    # ----------------------------------------------- #

    parser = argparse.ArgumentParser(
        description="Train SkinSegAI lesion segmentation model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

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
        help="Soft Dice loss weight (default: 0.4).",
    )

    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.3,
        help="BCE loss weight (default: 0.3).",
    )

    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.3,
        help="Boundary loss weight (default: 0.3).",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without ImageNet encoder initialization.",
    )

    args = parser.parse_args()

    # ----------------------------------------------- #
    # Validate training configuration
    # ----------------------------------------------- #

    if args.epochs < 1 or args.batch_size < 1:
        parser.error(
            "Epochs and batch size must be positive."
        )

    loss_weights = (
        args.dice_weight,
        args.bce_weight,
        args.boundary_weight,
    )

    if any(not math.isfinite(w) for w in loss_weights):
        parser.error(
            "Loss weights must be finite."
        )

    if any(w < 0 for w in loss_weights):
        parser.error(
            "Loss weights must be non-negative."
        )

    if sum(loss_weights) == 0:
        parser.error(
            "At least one loss weight must be positive."
        )

    # ----------------------------------------------- #
    # Reproducibility
    # ----------------------------------------------- #

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ----------------------------------------------- #
    # Project paths
    # ----------------------------------------------- #

    root = Path(__file__).resolve().parents[1]

    splits = root / "data/splits_grouped"

    registry = ModelArtifactRegistry(root)

    # ----------------------------------------------- #
    # Hardware selection
    # ----------------------------------------------- #

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ----------------------------------------------- #
    # Architecture-specific normalization
    # ----------------------------------------------- #

    imagenet_normalization = (
        uses_imagenet_normalization(args.arch)
    )

    if imagenet_normalization:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = None
        std = None

    pretrained = (
        args.arch != "custom_unet"
        and not args.no_pretrained
    )

    # ----------------------------------------------- #
    # Training dataset
    # ----------------------------------------------- #

    train_dataset = LesionDataset(
        csv_path=splits / "train.csv",
        project_root=root,
        augment=True,
        mean=mean,
        std=std,
    )

    # ----------------------------------------------- #
    # Validation dataset
    # ----------------------------------------------- #

    val_dataset = LesionDataset(
        csv_path=splits / "validation.csv",
        project_root=root,
        augment=False,
        mean=mean,
        std=std,
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    # ----------------------------------------------- #
    # Smoke-test configuration
    # ----------------------------------------------- #

    epochs = args.epochs

    if args.smoke:

        train_dataset = Subset(
            train_dataset,
            range(min(8, len(train_dataset))),
        )

        val_dataset = Subset(
            val_dataset,
            range(min(4, len(val_dataset))),
        )

        epochs = 1

    # ----------------------------------------------- #
    # Data loaders
    # ----------------------------------------------- #

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ----------------------------------------------- #
    # Initialize selected architecture
    # ----------------------------------------------- #

    model = build_model(
        args.arch,
        pretrained=pretrained,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    # ----------------------------------------------- #
    # Create isolated training directory
    # ----------------------------------------------- #

    output_dir = registry.create_run_directory(
        args.arch,
        kind="training",
        smoke=args.smoke,
    )

    # ----------------------------------------------- #
    # Initialize model trainer
    # ----------------------------------------------- #

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

    # ----------------------------------------------- #
    # Display training configuration
    # ----------------------------------------------- #

    print("\n===== TRAINING CONFIGURATION =====")

    print("Architecture:", args.arch)

    print(
        "Model:",
        registry.display_name(args.arch),
    )

    print("Device:", device)

    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    print("Batch size:", args.batch_size)
    print("Epochs:", epochs)

    print(
        "ImageNet normalization:",
        imagenet_normalization,
    )

    print(
        "Pretrained encoder initialization:",
        pretrained,
    )

    print(
        "Loss weights:",
        f"Dice={args.dice_weight}, "
        f"BCE={args.bce_weight}, "
        f"Boundary={args.boundary_weight}",
    )

    print("Smoke test:", args.smoke)

    print("Random seed: 42")

    print("Output directory:", output_dir)

    # ----------------------------------------------- #
    # Execute training
    # ----------------------------------------------- #

    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
    )

    # ----------------------------------------------- #
    # Verify generated artifacts
    # ----------------------------------------------- #

    checkpoint_path = (
        output_dir / "best_model.pth"
    )

    history_path = (
        output_dir / "training_history.csv"
    )

    if (
        not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Training completed without a valid checkpoint."
        )

    if (
        not history_path.is_file()
        or history_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Training completed without training history."
        )

    # ----------------------------------------------- #
    # Display final training status
    # ----------------------------------------------- #

    print("\n===== TRAINING STATUS =====")

    print("Architecture:", args.arch)
    print("Checkpoint generation: PASSED")
    print("Training history generation: PASSED")
    print("Experiment directory isolation: PASSED")

    print("\nGenerated artifacts:")

    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)

    print(
        "\nExisting production checkpoints "
        "have not been overwritten."
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
