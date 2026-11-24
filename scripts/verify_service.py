
import csv
import io
import zipfile

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.segmentation.unet_segmenter import UNetSegmenter
from src.segmentation.grabcut_segmenter import GrabCutSegmenter

from src.services.segmentation_service import SegmentationService
from src.services.report_exporter import ReportExporter


def main():

    root = Path(__file__).resolve().parents[1]

    cv2.setRNGSeed(42)

    service = SegmentationService([
        UNetSegmenter(
            root / "artifacts/checkpoints/best_model.pth"
        ),
        GrabCutSegmenter()
    ])

    exporter = ReportExporter()

    print(
        "Available algorithms:",
        service.available_algorithms
    )

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

    results = service.compare(
        image,
        ground_truth=ground_truth
    )

    assert len(results) == 2

    for result in results:

        assert result.mask.shape == ground_truth.shape
        assert result.metrics is not None

        print(f"\nAlgorithm: {result.algorithm}")

        print(
            f"Inference time: "
            f"{result.inference_ms:.2f} ms"
        )

        print(
            f"Lesion coverage: "
            f"{result.measurements['area_percent']:.2f}%"
        )

        print(f"Dice: {result.metrics['dice']:.4f}")
        print(f"IoU: {result.metrics['iou']:.4f}")

    single_result = service.segment(image, "U-Net")

    assert single_result.metrics is None

    print("\nInference without ground truth: PASSED")

    report = exporter.export_zip(results)

    with zipfile.ZipFile(io.BytesIO(report)) as archive:

        assert archive.testzip() is None
        assert len(archive.namelist()) == 10

        print("ZIP integrity: PASSED")

    output = (
        root / "artifacts/exports/backend_report.zip"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_bytes(report)

    print("\nReport export: PASSED")
    print("Report saved to:", output)

    print("\nComplete backend verification successful!")


if __name__ == "__main__":
    main()
