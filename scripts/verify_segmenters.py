
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.segmentation.unet_segmenter import UNetSegmenter
from src.segmentation.grabcut_segmenter import GrabCutSegmenter
from src.evaluation.metrics import MetricsEvaluator


def main():

    root = Path(__file__).resolve().parents[1]

    cv2.setRNGSeed(42)

    segmenters = [
        UNetSegmenter(
            root / "artifacts/checkpoints/best_model.pth"
        ),
        GrabCutSegmenter(iterations=5)
    ]

    evaluator = MetricsEvaluator()

    with open(
        root / "data/splits_grouped/validation.csv",
        newline=""
    ) as file:
        record = next(csv.DictReader(file))

    with Image.open(root / record["image"]) as source:
        image = source.convert("RGB")

    with Image.open(root / record["mask"]) as source:
        ground_truth = np.asarray(
            source.convert("L")
        ) > 127

    print("\n===== SEGMENTATION VERIFICATION =====")

    for segmenter in segmenters:

        prediction = segmenter.segment(image)

        assert prediction.shape == ground_truth.shape
        assert prediction.dtype == np.bool_

        metrics = evaluator.evaluate_masks(
            torch.from_numpy(prediction).unsqueeze(0),
            torch.from_numpy(ground_truth).unsqueeze(0)
        )

        print(f"\nAlgorithm: {segmenter.name}")
        print("Mask shape:", prediction.shape)
        print("Mask type:", prediction.dtype)

        print(
            f"Affected image area: "
            f"{prediction.mean() * 100:.2f}%"
        )

        for name, value in metrics.items():
            print(f"{name.upper()}: {value:.4f}")

    print("\nBoth segmentation algorithms verified!")


if __name__ == "__main__":
    main()
