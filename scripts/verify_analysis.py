
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.analysis.lesion_analyzer import LesionAnalyzer

from src.segmentation.unet_segmenter import UNetSegmenter

from src.segmentation.grabcut_segmenter import GrabCutSegmenter


def main():

    root = Path(__file__).resolve().parents[1]

    analyzer = LesionAnalyzer()

    # Verify measurements on a known synthetic mask.
    square = np.ones((4, 4), dtype=bool)
    result = analyzer.analyze(square)

    assert result["area_pixels"] == 16
    assert result["perimeter_pixels"] == 16
    assert result["asymmetry"] == 0.0
    assert result["area_percent"] == 100.0

    empty = np.zeros((4, 4), dtype=bool)
    assert analyzer.analyze(empty)["area_pixels"] == 0

    print("Synthetic measurement tests: PASSED")

    segmenters = [

        UNetSegmenter(
            root / "artifacts/checkpoints/best_model.pth"
        ),

        GrabCutSegmenter()
    ]

    with open(
        root / "data/splits_grouped/validation.csv",
        newline=""
    ) as file:

        record = next(csv.DictReader(file))

    image_id = Path(record["image"]).stem

    image_path = (
        root
        / "data/raw/ISIC2018_Task1-2_Training_Input"
        / f"{image_id}.jpg"
    )

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    output_dir = (
        root / "artifacts/exports/verification"
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    print("\n===== LESION ANALYSIS =====")
    print("Original resolution:", image.size)

    image.save(output_dir / "original.png")

    for segmenter in segmenters:

        mask = segmenter.segment(image)

        results = analyzer.analyze(mask)

        overlay = analyzer.create_overlay(
            image, mask
        )

        boundary = analyzer.draw_boundary(
            image, mask
        )

        name = segmenter.name.lower().replace("-", "")

        Image.fromarray(
            mask.astype(np.uint8) * 255
        ).save(output_dir / f"{name}_mask.png")

        overlay.save(
            output_dir / f"{name}_overlay.png"
        )

        boundary.save(
            output_dir / f"{name}_boundary.png"
        )

        with open(
            output_dir / f"{name}_analysis.json",
            "w"
        ) as file:

            json.dump(results, file, indent=4)

        print(f"\nAlgorithm: {segmenter.name}")
        print(json.dumps(results, indent=4))

    print("\nLesion analysis verification successful!")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
