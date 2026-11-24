
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.lesion_dataset import LesionDataset
from src.models.unet import UNet
from src.training.model_trainer import ModelTrainer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]
    splits = root / "data/splits_grouped"

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train_dataset = LesionDataset(
        csv_path=splits / "train.csv",
        project_root=root,
        augment=True
    )

    val_dataset = LesionDataset(
        csv_path=splits / "validation.csv",
        project_root=root,
        augment=False
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

    model = UNet()

    trainer = ModelTrainer(
        model=model,
        device=device,
        output_dir=output_dir,
        learning_rate=0.001,
        patience=7
    )

    print("\n===== TRAINING CONFIGURATION =====")
    print("Device:", device)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", epochs)
    print("Smoke test:", args.smoke)

    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs
    )


if __name__ == "__main__":
    main()
