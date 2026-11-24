
from pathlib import Path

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.lesion_dataset import LesionDataset
from src.models.unet import UNet


def main():

    random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Selected device:", device)

    dataset = LesionDataset(
        csv_path=root / "data/splits/train.csv",
        project_root=root,
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    images, masks = next(iter(loader))

    print("Training samples:", len(dataset))
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)

    assert images.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 1, 256, 256)
    assert torch.all((masks == 0) | (masks == 1))

    images = images.to(device)
    masks = masks.to(device)

    model = UNet().to(device)
    model.train()

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", parameters)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    predictions = model(images)

    print("Output shape:", predictions.shape)

    assert predictions.shape == masks.shape

    loss = criterion(predictions, masks)

    assert torch.isfinite(loss).item()

    loss.backward()
    optimizer.step()

    print("Test loss:", loss.item())
    print("Forward pass: PASSED")
    print("Backward pass: PASSED")
    print("Optimizer step: PASSED")

    print("\nU-Net verification successful!")


if __name__ == "__main__":
    main()
