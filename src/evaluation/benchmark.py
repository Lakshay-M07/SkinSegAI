
import time

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from src.evaluation.metrics import MetricsEvaluator
from src.core.base_segmenter import BaseSegmenter


class SegmentationBenchmark:

    def __init__(self, segmenters):

        if not segmenters:
            raise ValueError("No segmenters provided.")

        if not all(
            isinstance(s, BaseSegmenter)
            for s in segmenters
        ):
            raise TypeError("Invalid segmenter.")

        self._segmenters = segmenters
        self._evaluator = MetricsEvaluator()

    def evaluate(self, records, project_root):

        results = []

        for record in tqdm(
            records, desc="Evaluating test dataset"
        ):

            with Image.open(
                project_root / record["image"]
            ) as source:

                image = source.convert("RGB")

            with Image.open(
                project_root / record["mask"]
            ) as source:

                ground_truth = np.asarray(
                    source.convert("L")
                ) > 127

            for segmenter in self._segmenters:

                cv2.setRNGSeed(42)

                start = time.perf_counter()

                prediction = segmenter.segment(image)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                if prediction.shape != ground_truth.shape:
                    raise ValueError(
                        "Prediction dimensions mismatch."
                    )

                if prediction.dtype != np.bool_:
                    raise TypeError("Expected binary mask.")

                metrics = self._evaluator.evaluate_masks(
                    torch.from_numpy(
                        prediction
                    ).unsqueeze(0),

                    torch.from_numpy(
                        ground_truth
                    ).unsqueeze(0)
                )

                results.append({
                    "image": record["image"],
                    "algorithm": segmenter.name,
                    **metrics,
                    "inference_ms": elapsed_ms,
                    "predicted_area_percent":
                        prediction.mean() * 100,
                    "actual_area_percent":
                        ground_truth.mean() * 100
                })

        return pd.DataFrame(results)

    @staticmethod
    def summarize(results):

        return results.groupby("algorithm").agg(

            samples=("dice", "count"),

            dice=("dice", "mean"),
            dice_std=("dice", "std"),

            iou=("iou", "mean"),

            precision=("precision", "mean"),

            recall=("recall", "mean"),

            inference_ms=("inference_ms", "mean"),

            median_inference_ms=(
                "inference_ms", "median"
            )

        ).reset_index()
