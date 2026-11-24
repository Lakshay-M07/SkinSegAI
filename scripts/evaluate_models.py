
import csv
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset_manager import DatasetManager
from src.data.dataset_splitter import DatasetSplitter
from src.data.metadata_loader import MetadataLoader

from src.segmentation.unet_segmenter import UNetSegmenter
from src.segmentation.grabcut_segmenter import GrabCutSegmenter

from src.evaluation.benchmark import SegmentationBenchmark


def read_records(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}"
        )

    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def verify_splits(root):

    print("\n===== DATASET VERIFICATION =====")

    manager = DatasetManager(root / "data/raw")
    manager.load_dataset()

    # Automatically discover the metadata CSV.
    metadata_loader = MetadataLoader(root / "data/raw")
    metadata_path = metadata_loader.find_metadata()

    print("Metadata file:", metadata_path.name)

    splitter = DatasetSplitter(
        metadata_path=metadata_path,
        random_seed=42
    )

    expected = splitter.split(manager.get_pairs())

    for name, pairs in expected.items():

        records = read_records(
            root / "data/splits_grouped" / f"{name}.csv"
        )

        actual_ids = [
            Path(record["image"]).stem
            for record in records
        ]

        expected_ids = {
            image.stem for image, _ in pairs
        }

        if (
            set(actual_ids) != expected_ids
            or len(actual_ids) != len(expected_ids)
        ):
            raise ValueError(
                f"Split verification failed: {name}"
            )

        for record in records:

            image_path = root / record["image"]
            mask_path = root / record["mask"]

            image_id = image_path.stem
            mask_id = mask_path.stem

            if mask_id != image_id + "_segmentation":
                raise ValueError(
                    f"Image-mask ID mismatch: {image_id}"
                )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing processed image: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing processed mask: {mask_path}"
                )

        print(
            f"{name.capitalize()} split: "
            f"{len(records)} samples - PASSED"
        )

    print("\nGrouped dataset manifests verified.")


def main():

    root = Path(__file__).resolve().parents[1]

    # Verify dataset integrity and group separation.
    verify_splits(root)

    # Select Apple GPU when available.
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint_path = (
        root / "artifacts/checkpoints/best_model.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: "
            f"{checkpoint_path}"
        )

    # Initialize both segmentation algorithms.
    segmenters = [

        UNetSegmenter(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=0.5
        ),

        GrabCutSegmenter(
            iterations=5,
            margin=0.05
        )
    ]

    # Load the independent test dataset.
    records = read_records(
        root / "data/splits_grouped/test.csv"
    )

    if not records:
        raise ValueError("Test dataset is empty.")

    # Warm up both algorithms using a validation image.
    validation = read_records(
        root / "data/splits_grouped/validation.csv"
    )

    if not validation:
        raise ValueError("Validation dataset is empty.")

    with Image.open(
        root / validation[0]["image"]
    ) as source:

        warmup_image = source.convert("RGB")

    print("\nWarming up segmentation algorithms...")

    for segmenter in segmenters:

        segmenter.segment(warmup_image)

        print(f"{segmenter.name}: Ready")

    # Display benchmark configuration.
    print("\n===== BENCHMARK CONFIGURATION =====")

    print("Test samples:", len(records))
    print("U-Net device:", device)
    print("GrabCut device: CPU")
    print("U-Net threshold: 0.5")
    print("GrabCut iterations: 5")
    print("GrabCut margin: 0.05")
    print("Model checkpoint:", checkpoint_path.name)

    # Execute comparative benchmarking.
    benchmark = SegmentationBenchmark(segmenters)

    results = benchmark.evaluate(records, root)

    summary = benchmark.summarize(results)

    # Verify that both algorithms evaluated all samples.
    expected_rows = len(records) * len(segmenters)

    if len(results) != expected_rows:
        raise RuntimeError(
            f"Incomplete benchmark results. "
            f"Expected {expected_rows}, "
            f"received {len(results)}."
        )

    for segmenter in segmenters:

        algorithm_results = results[
            results["algorithm"] == segmenter.name
        ]

        if (
            len(algorithm_results) != len(records)
            or algorithm_results["image"].duplicated().any()
            or set(algorithm_results["image"]) != {
                record["image"] for record in records
            }
        ):
            raise RuntimeError(
                f"Incomplete or duplicate results for "
                f"{segmenter.name}."
            )

    # Create the results directory.
    output_dir = root / "artifacts/metrics"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save individual image results.
    results_path = (
        output_dir / "per_image_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # Save aggregated comparative results.
    summary_path = (
        output_dir / "comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # Display the final comparison table.
    print("\n===== FINAL COMPARISON =====")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n===== EVALUATION STATUS =====")

    print("Dataset split verification: PASSED")
    print("Model checkpoint loading: PASSED")
    print("Segmentation benchmarking: PASSED")
    print("Results completeness: PASSED")
    print("CSV report generation: PASSED")

    print("\nEvaluation completed successfully.")

    print("Results saved to:", output_dir)

    print("\nGenerated files:")

    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
