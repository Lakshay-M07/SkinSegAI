
"""Evaluate trained SkinSegAI segmentation models."""

import argparse
import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from PIL import Image

from src.data.dataset_manager import DatasetManager
from src.data.dataset_splitter import DatasetSplitter
from src.data.metadata_loader import MetadataLoader

from src.models.artifacts import ModelArtifactRegistry
from src.models.model_factory import SUPPORTED_ARCHITECTURES

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

    metadata_loader = MetadataLoader(root / "data/raw")
    metadata_path = metadata_loader.find_metadata()

    print("Metadata file:", metadata_path.name)

    splitter = DatasetSplitter(
        metadata_path=metadata_path,
        random_seed=42,
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

    # ----------------------------------------------- #
    # Command-line configuration
    # ----------------------------------------------- #

    parser = argparse.ArgumentParser(
        description="Evaluate SkinSegAI segmentation models."
    )

    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="resnet34_unet",
        help="Architecture to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint from a new training run.",
    )

    args = parser.parse_args()

    registry = ModelArtifactRegistry(root)

    # ----------------------------------------------- #
    # Resolve checkpoint
    # ----------------------------------------------- #

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else registry.require_checkpoint(args.arch)
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: "
            f"{checkpoint_path}"
        )

    # ----------------------------------------------- #
    # Verify dataset
    # ----------------------------------------------- #

    verify_splits(root)

    # ----------------------------------------------- #
    # Hardware selection
    # ----------------------------------------------- #

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ----------------------------------------------- #
    # Initialize segmentation algorithms
    # ----------------------------------------------- #

    unet = UNetSegmenter(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=0.5,
    )

    if unet.arch != args.arch:
        raise ValueError(
            f"Checkpoint architecture mismatch: "
            f"expected '{args.arch}', "
            f"found '{unet.arch}'."
        )

    grabcut = GrabCutSegmenter(
        iterations=5,
        margin=0.05,
    )

    segmenters = [unet, grabcut]

    # ----------------------------------------------- #
    # Load independent test dataset
    # ----------------------------------------------- #

    records = read_records(
        root / "data/splits_grouped/test.csv"
    )

    if not records:
        raise ValueError("Test dataset is empty.")

    validation = read_records(
        root / "data/splits_grouped/validation.csv"
    )

    if not validation:
        raise ValueError("Validation dataset is empty.")

    # ----------------------------------------------- #
    # Warm up both algorithms
    # ----------------------------------------------- #

    with Image.open(
        root / validation[0]["image"]
    ) as source:

        warmup_image = source.convert("RGB")

    print("\nWarming up segmentation algorithms...")

    for segmenter in segmenters:

        segmenter.segment(warmup_image)

        print(f"{segmenter.name}: Ready")

    if device == "cuda":
        torch.cuda.synchronize()

    elif device == "mps":
        torch.mps.synchronize()

    # ----------------------------------------------- #
    # Display benchmark configuration
    # ----------------------------------------------- #

    print("\n===== BENCHMARK CONFIGURATION =====")

    print("Architecture:", args.arch)
    print("Model:", registry.display_name(args.arch))
    print("Test samples:", len(records))
    print("U-Net device:", device)
    print("GrabCut device: CPU")
    print("U-Net threshold: 0.5")
    print("GrabCut iterations: 5")
    print("GrabCut margin: 0.05")
    print("Model checkpoint:", checkpoint_path)

    # ----------------------------------------------- #
    # Execute benchmark
    # ----------------------------------------------- #

    benchmark = SegmentationBenchmark(segmenters)

    results = benchmark.evaluate(
        records,
        root,
    )

    # ----------------------------------------------- #
    # Verify result completeness
    # ----------------------------------------------- #

    expected_rows = len(records) * len(segmenters)

    if len(results) != expected_rows:
        raise RuntimeError(
            f"Incomplete benchmark results. "
            f"Expected {expected_rows}, "
            f"received {len(results)}."
        )

    expected_images = {
        record["image"]
        for record in records
    }

    for segmenter in segmenters:

        algorithm_results = results[
            results["algorithm"] == segmenter.name
        ]

        if (
            len(algorithm_results) != len(records)
            or algorithm_results["image"].duplicated().any()
            or set(algorithm_results["image"]) != expected_images
        ):
            raise RuntimeError(
                f"Incomplete or duplicate results for "
                f"{segmenter.name}."
            )

    summary = benchmark.summarize(results)

    # ----------------------------------------------- #
    # Create isolated evaluation directory
    # ----------------------------------------------- #

    output_dir = registry.create_run_directory(
        args.arch,
        kind="evaluation",
    )

    # ----------------------------------------------- #
    # Export individual image results
    # ----------------------------------------------- #

    results_path = (
        output_dir / "per_image_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # ----------------------------------------------- #
    # Export aggregated comparison
    # ----------------------------------------------- #

    summary_path = (
        output_dir / "comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ----------------------------------------------- #
    # Display results
    # ----------------------------------------------- #

    print("\n===== FINAL COMPARISON =====")

    print("Architecture:", args.arch)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n===== EVALUATION STATUS =====")

    print("Dataset split verification: PASSED")
    print("Model checkpoint loading: PASSED")
    print("Architecture verification: PASSED")
    print("Segmentation benchmarking: PASSED")
    print("Results completeness: PASSED")
    print("CSV report generation: PASSED")
    print("Experiment directory isolation: PASSED")

    print("\nEvaluation completed successfully.")

    print("Results saved to:", output_dir)

    print("\nGenerated files:")

    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
