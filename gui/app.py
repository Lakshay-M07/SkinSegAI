
import csv
import hashlib
import io
import sys
import logging

from pathlib import Path
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from PIL import Image, UnidentifiedImageError


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.segmentation.unet_segmenter import UNetSegmenter
from src.segmentation.grabcut_segmenter import GrabCutSegmenter

from src.services.segmentation_service import SegmentationService
from src.services.report_exporter import ReportExporter

from src.analysis.lesion_analyzer import LesionAnalyzer


logger = logging.getLogger(__name__)


# ============================================================
# MAIN APPLICATION CLASS
# ============================================================

class SkinSegApp:

    MAX_UPLOAD_BYTES = 20 * 1024 * 1024
    MAX_IMAGE_PIXELS = 12_000_000

    def __init__(self):

        self._root = ROOT

        self._checkpoint = (
            ROOT / "artifacts/checkpoints/best_model.pth"
        )

        self._summary_path = (
            ROOT / "artifacts/metrics/comparison_summary.csv"
        )

        self._results_path = (
            ROOT / "artifacts/metrics/per_image_results.csv"
        )

        self._history_path = (
            ROOT / "artifacts/checkpoints/training_history.csv"
        )

        self._exporter = ReportExporter()

        self._analyzer = LesionAnalyzer()

        self._colors = {
            "U-Net": "#38BDF8",
            "GrabCut": "#2DD4BF"
        }

    # ========================================================
    # BACKEND SERVICE
    # ========================================================

    def _get_service(self, threshold):

        if "segmentation_service" not in st.session_state:

            unet = UNetSegmenter(
                self._checkpoint,
                threshold=threshold
            )

            grabcut = GrabCutSegmenter()

            service = SegmentationService(
                [unet, grabcut]
            )

            st.session_state["unet_segmenter"] = unet

            st.session_state[
                "segmentation_service"
            ] = service

        st.session_state[
            "unet_segmenter"
        ].threshold = threshold

        return st.session_state[
            "segmentation_service"
        ]

    # ========================================================
    # CSV LOADING
    # ========================================================

    @staticmethod
    def _load_csv(path):

        if not path.is_file():

            st.error(
                f"Required file missing: {path.name}"
            )

            return None

        try:

            dataframe = pd.read_csv(path)

            if dataframe.empty:

                st.warning(
                    f"The file {path.name} is empty."
                )

                return None

            return dataframe

        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError
        ):

            logger.exception("CSV loading failed.")

            st.error(
                f"Unable to read {path.name}. "
                "Please verify the generated results file."
            )

            return None

    # ========================================================
    # IMAGE DECODING AND VALIDATION
    # UPDATED: FRIENDLY ERROR HANDLING
    # ========================================================

    @staticmethod
    def _decode_image(data, mode="RGB"):

        file_type = (
            "ground-truth mask"
            if mode == "L"
            else "image"
        )

        if not data:

            raise ValueError(
                f"The uploaded {file_type} is empty."
            )

        if len(data) > SkinSegApp.MAX_UPLOAD_BYTES:

            raise ValueError(
                "File exceeds the 20 MB upload limit."
            )

        try:

            with Image.open(io.BytesIO(data)) as image:

                if image.format not in {"JPEG", "PNG"}:

                    raise ValueError(
                        "Unsupported file format. "
                        "Upload a JPEG or PNG file."
                    )

                if mode == "L" and image.format != "PNG":

                    raise ValueError(
                        "Ground-truth masks must be PNG files."
                    )

                if (
                    image.width * image.height
                    > SkinSegApp.MAX_IMAGE_PIXELS
                ):

                    raise ValueError(
                        "Image exceeds the 12-megapixel "
                        "resolution limit. "
                        "Please upload a smaller image."
                    )

                # Force complete image decoding to detect
                # damaged or truncated image contents.
                image.load()

                return image.convert(mode)

        except Image.DecompressionBombError:

            raise ValueError(
                "Image dimensions are too large. "
                "Upload a smaller image."
            ) from None

        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError
        ):

            raise ValueError(
                f"This {file_type} could not be opened. "
                + (
                    "Upload a readable PNG mask."
                    if mode == "L"
                    else "Upload a readable JPEG or PNG."
                )
            ) from None

    # ========================================================
    # DASHBOARD
    # ========================================================

    def _dashboard(self):

        st.title("SkinSegAI Dashboard")

        st.caption(
            "Deep learning-based skin lesion "
            "segmentation and comparative analysis."
        )

        summary = self._load_csv(self._summary_path)

        if summary is None:
            return

        required_columns = {
            "algorithm",
            "samples",
            "dice",
            "iou",
            "precision",
            "recall",
            "inference_ms"
        }

        if not required_columns.issubset(summary.columns):

            st.error(
                "The benchmark summary is incomplete. "
                "Run the model evaluation script again."
            )

            return

        unet_rows = summary.loc[
            summary["algorithm"] == "U-Net"
        ]

        grabcut_rows = summary.loc[
            summary["algorithm"] == "GrabCut"
        ]

        if len(unet_rows) != 1 or len(grabcut_rows) != 1:

            st.error(
                "Benchmark results for both algorithms "
                "are required."
            )

            return

        unet = unet_rows.iloc[0]

        grabcut = grabcut_rows.iloc[0]

        cols = st.columns(4)

        cols[0].metric(
            "Test Dice",
            f"{unet['dice']:.2%}"
        )

        cols[1].metric(
            "Test IoU",
            f"{unet['iou']:.2%}"
        )

        cols[2].metric(
            "Mean inference",
            f"{unet['inference_ms']:.2f} ms"
        )

        cols[3].metric(
            "Test images",
            int(unet["samples"])
        )

        st.divider()

        st.subheader("Algorithm performance")

        chart_data = summary.melt(
            id_vars="algorithm",
            value_vars=[
                "dice",
                "iou",
                "precision",
                "recall"
            ],
            var_name="metric",
            value_name="score"
        )

        chart = px.bar(
            chart_data,
            x="metric",
            y="score",
            color="algorithm",
            barmode="group",
            color_discrete_map=self._colors,
            text_auto=".3f",
            labels={
                "metric": "Evaluation metric",
                "score": "Score",
                "algorithm": "Algorithm"
            }
        )

        chart.update_yaxes(range=[0, 1.05])

        st.plotly_chart(chart)

        improvement = (
            unet["dice"] - grabcut["dice"]
        ) * 100

        speedup = (
            grabcut["inference_ms"]
            / unet["inference_ms"]
            if unet["inference_ms"] > 0
            else None
        )

        left, right = st.columns(2)

        left.metric(
            "Dice improvement",
            f"{improvement:.2f} percentage points"
        )

        right.metric(
            "Observed inference speedup",
            f"{speedup:.1f}x"
            if speedup is not None
            else "N/A"
        )

        st.caption(
            "U-Net was evaluated on the Apple GPU, "
            "while GrabCut was evaluated on the CPU "
            "in the original benchmark."
        )

        st.divider()

        st.subheader("System information")

        st.write({
            "Dataset": "ISIC 2018 Task 1",
            "Total dataset images": 2594,
            "Trainable parameters": 1942577,
            "Model input resolution": "256 x 256",
            "Primary model": "Custom U-Net",
            "Baseline algorithm": "GrabCut",
            "Framework": "PyTorch"
        })

        st.info(
            "SkinSegAI is intended for academic "
            "research and educational demonstrations. "
            "It is not a medical diagnostic system."
        )

    # ========================================================
    # SEGMENTATION RESULT RENDERING
    # ========================================================

    def _render_result(
        self,
        result,
        original,
        opacity
    ):

        st.subheader(result.algorithm)

        updated_overlay = self._analyzer.create_overlay(
            original,
            result.mask,
            alpha=opacity
        )

        result = replace(
            result,
            overlay=updated_overlay
        )

        tabs = st.tabs([
            "Overlay",
            "Binary Mask",
            "Boundary"
        ])

        with tabs[0]:

            st.image(
                result.overlay,
                caption="Segmentation overlay"
            )

        with tabs[1]:

            st.image(
                result.mask.astype(np.uint8) * 255,
                caption="Predicted binary mask"
            )

        with tabs[2]:

            st.image(
                result.boundary,
                caption="Detected lesion boundary"
            )

        measurements = result.measurements

        left, right = st.columns(2)

        left.metric(
            "Lesion coverage",
            f"{measurements['area_percent']:.2f}%"
        )

        right.metric(
            "Inference time",
            f"{result.inference_ms:.2f} ms"
        )

        if measurements["area_pixels"] == 0:

            st.warning(
                "No lesion pixels were predicted. "
                "This does not establish that the "
                "image contains no lesion."
            )

        elif measurements["area_percent"] == 100:

            st.warning(
                "The entire image was classified "
                "as foreground."
            )

        circularity = measurements["circularity"]

        asymmetry = measurements["asymmetry"]

        table = pd.DataFrame({

            "Measurement": [
                "Lesion area (px²)",
                "Perimeter (px)",
                "Circularity",
                "Asymmetry"
            ],

            "Value": [

                str(measurements["area_pixels"]),

                f"{measurements['perimeter_pixels']:.2f}",

                f"{circularity:.4f}"
                if circularity is not None
                else "N/A",

                f"{asymmetry:.4f}"
                if asymmetry is not None
                else "N/A"
            ]
        })

        st.dataframe(
            table,
            hide_index=True
        )

        with st.expander("Geometrical details"):

            st.json({

                "bounding_box":
                    measurements["bounding_box"],

                "centroid":
                    measurements["centroid"]
            })

        if result.metrics is not None:

            st.markdown(
                "**Ground-truth evaluation**"
            )

            for name, value in result.metrics.items():

                st.write(
                    f"{name.upper()}: {value:.4f}"
                )

                st.progress(
                    float(np.clip(value, 0, 1))
                )

        return result

    # ========================================================
    # LIVE SEGMENTATION PAGE
    # ========================================================

    def _segmentation(self):

        st.title("Live Lesion Segmentation")

        st.caption(
            "Upload a dermoscopic image or explore "
            "a validation dataset sample."
        )

        st.info(
            "Experimental segmentation only. "
            "The predicted mask cannot determine "
            "whether a lesion is benign or malignant."
        )

        source = st.radio(
            "Image source",
            ["Upload image", "Dataset sample"],
            horizontal=True
        )

        image_bytes = None

        mask_bytes = None

        # ----------------------------------------------------
        # USER UPLOAD
        # ----------------------------------------------------

        if source == "Upload image":

            uploaded = st.file_uploader(
                "Upload dermoscopic image",
                type=["jpg", "jpeg", "png"],
                key="image_upload"
            )

            ground_truth_file = st.file_uploader(
                "Ground-truth mask (optional)",
                type=["png"],
                key="mask_upload"
            )

            st.caption(
                "Supported formats: JPEG and PNG. "
                "Maximum file size: 20 MB. "
                "Maximum resolution: 12 megapixels."
            )

            if uploaded is not None:

                image_bytes = uploaded.getvalue()

            if ground_truth_file is not None:

                mask_bytes = (
                    ground_truth_file.getvalue()
                )

        # ----------------------------------------------------
        # VALIDATION DATASET SAMPLE
        # ----------------------------------------------------

        else:

            manifest = (
                self._root /
                "data/splits_grouped/validation.csv"
            )

            if not manifest.is_file():

                st.error(
                    "Validation manifest missing."
                )

                return

            try:

                with open(
                    manifest,
                    newline="",
                    encoding="utf-8"
                ) as file:

                    records = list(
                        csv.DictReader(file)
                    )

            except (
                OSError,
                UnicodeError,
                csv.Error
            ):

                logger.exception(
                    "Validation manifest loading failed."
                )

                st.error(
                    "Unable to read the validation dataset."
                )

                return

            if not records:

                st.error(
                    "Validation dataset is empty."
                )

                return

            if any(
                not record.get("image")
                or not record.get("mask")
                for record in records
            ):

                st.error(
                    "The validation manifest "
                    "contains incomplete records."
                )

                return

            selected = st.selectbox(
                "Select validation image",
                range(len(records)),
                format_func=lambda i: Path(
                    records[i]["image"]
                ).stem
            )

            record = records[selected]

            try:

                image_bytes = (
                    self._root / record["image"]
                ).read_bytes()

                mask_bytes = (
                    self._root / record["mask"]
                ).read_bytes()

            except OSError:

                logger.exception(
                    "Dataset sample loading failed."
                )

                st.error(
                    "The selected dataset image or mask "
                    "could not be loaded. "
                    "Verify that the processed dataset exists."
                )

                return

            st.caption(
                "Dataset samples use the preprocessed "
                "256 x 256 validation images."
            )

        # ----------------------------------------------------
        # SEGMENTATION SETTINGS
        # ----------------------------------------------------

        st.divider()

        algorithm = st.radio(
            "Segmentation algorithm",
            [
                "U-Net",
                "GrabCut",
                "Compare both"
            ],
            horizontal=True
        )

        threshold = st.slider(
            "U-Net prediction threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            disabled=(algorithm == "GrabCut"),
            help=(
                "Controls the probability threshold "
                "used to classify lesion pixels."
            )
        )

        opacity = st.slider(
            "Overlay opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )

        if image_bytes is None:

            st.info(
                "Select an image to begin."
            )

            return

        # ----------------------------------------------------
        # IMAGE AND MASK VALIDATION
        # ----------------------------------------------------

        try:

            image = self._decode_image(
                image_bytes
            )

            ground_truth = None

            if mask_bytes is not None:

                mask_image = self._decode_image(
                    mask_bytes,
                    mode="L"
                )

                if mask_image.size != image.size:

                    raise ValueError(
                        "Image and ground-truth mask "
                        "dimensions must match."
                    )

                array = np.asarray(mask_image)

                values = set(
                    np.unique(array).tolist()
                )

                if not (

                    values.issubset({0, 255})

                    or values.issubset({0, 1})

                ):

                    raise ValueError(
                        "Ground-truth mask must be binary. "
                        "Use a mask containing only "
                        "foreground and background pixels."
                    )

                ground_truth = array > 0

        except ValueError as error:

            st.session_state.pop(
                "segmentation_result",
                None
            )

            st.error(str(error))

            return

        except Exception:

            logger.exception(
                "Unexpected image processing error."
            )

            st.session_state.pop(
                "segmentation_result",
                None
            )

            st.error(
                "The uploaded file could not be processed. "
                "Please try another JPEG or PNG image."
            )

            return

        # ----------------------------------------------------
        # ORIGINAL IMAGE PREVIEW
        # ----------------------------------------------------

        st.subheader("Input image")

        preview = image.copy()

        preview.thumbnail((700, 700))

        st.image(
            preview,
            caption=(
                f"{image.width} x "
                f"{image.height} pixels"
            )
        )

        if ground_truth is not None:

            with st.expander(
                "View ground-truth mask"
            ):

                st.image(
                    ground_truth.astype(np.uint8) * 255
                )

        # ----------------------------------------------------
        # INPUT SIGNATURE FOR SESSION STATE
        # ----------------------------------------------------

        signature = (

            hashlib.sha256(
                image_bytes
            ).hexdigest(),

            hashlib.sha256(
                mask_bytes
            ).hexdigest()
            if mask_bytes is not None
            else None,

            algorithm,

            threshold
            if algorithm != "GrabCut"
            else None
        )

        # ----------------------------------------------------
        # EXECUTE SEGMENTATION
        # ----------------------------------------------------

        if st.button(
            "Run Segmentation",
            type="primary"
        ):

            st.session_state.pop(
                "segmentation_result",
                None
            )

            try:

                with st.spinner(
                    "Performing lesion segmentation..."
                ):

                    service = self._get_service(
                        threshold
                    )

                    cv2.setRNGSeed(42)

                    if algorithm == "Compare both":

                        results = service.compare(
                            image,
                            alpha=opacity,
                            ground_truth=ground_truth
                        )

                    else:

                        results = [

                            service.segment(
                                image,
                                algorithm,
                                alpha=opacity,
                                ground_truth=ground_truth
                            )
                        ]

                st.session_state[
                    "segmentation_result"
                ] = (
                    signature,
                    results
                )

            except FileNotFoundError:

                logger.exception(
                    "Model checkpoint or file missing."
                )

                st.error(
                    "A required model or data file is missing. "
                    "Verify that best_model.pth exists "
                    "in artifacts/checkpoints."
                )

            except Exception:

                logger.exception(
                    "Segmentation execution failed."
                )

                st.error(
                    "Segmentation could not be completed. "
                    "Try another image or check the "
                    "application terminal for details."
                )

        # ----------------------------------------------------
        # RETRIEVE SAVED RESULTS
        # ----------------------------------------------------

        saved = st.session_state.get(
            "segmentation_result"
        )

        if saved is None:
            return

        if saved[0] != signature:

            st.info(
                "Input or algorithm settings changed. "
                "Run segmentation again."
            )

            return

        results = saved[1]

        st.success(
            "Segmentation completed successfully."
        )

        st.divider()

        st.subheader("Segmentation results")

        columns = st.columns(
            len(results)
        )

        updated_results = []

        for column, result in zip(
            columns,
            results
        ):

            with column:

                updated_results.append(

                    self._render_result(
                        result,
                        image,
                        opacity
                    )
                )

        # ----------------------------------------------------
        # SIDE-BY-SIDE COMPARISON
        # ----------------------------------------------------

        if len(results) == 2:

            st.divider()

            st.subheader(
                "Side-by-side comparison"
            )

            comparison = pd.DataFrame([

                {

                    "Algorithm":
                        result.algorithm,

                    "Lesion coverage (%)":
                        result.measurements[
                            "area_percent"
                        ],

                    "Inference (ms)":
                        result.inference_ms,

                    **(result.metrics or {})
                }

                for result in results
            ])

            st.dataframe(
                comparison.round(4),
                hide_index=True
            )

        if ground_truth is None:

            st.info(
                "Dice, IoU, precision and recall "
                "require a corresponding "
                "ground-truth mask."
            )

        st.caption(
            "Lesion measurements are approximate "
            "pixel-based geometric measurements, "
            "not clinical diagnostic indicators."
        )

        # ----------------------------------------------------
        # REPORT DOWNLOAD
        # ----------------------------------------------------

        st.divider()

        try:

            report = self._exporter.export_zip(
                updated_results
            )

            st.download_button(
                label="Download Complete Analysis Report",
                data=report,
                file_name="SkinSegAI_Report.zip",
                mime="application/zip"
            )

        except Exception:

            logger.exception(
                "Report generation failed."
            )

            st.error(
                "The analysis report could not be generated. "
                "Please run segmentation again."
            )

    # ========================================================
    # MODEL COMPARISON PAGE
    # ========================================================

    def _comparison(self):

        st.title(
            "Comparative Performance Analysis"
        )

        summary = self._load_csv(
            self._summary_path
        )

        results = self._load_csv(
            self._results_path
        )

        if summary is None or results is None:
            return

        required_summary = {
            "algorithm",
            "dice",
            "iou",
            "precision",
            "recall",
            "inference_ms"
        }

        required_results = {
            "algorithm",
            "dice",
            "iou",
            "precision",
            "recall"
        }

        if not (
            required_summary.issubset(summary.columns)
            and required_results.issubset(results.columns)
        ):

            st.error(
                "Benchmark results are incomplete. "
                "Run the evaluation script again."
            )

            return

        st.caption(
            "Evaluation of both algorithms on "
            "the same held-out test dataset."
        )

        st.dataframe(
            summary.round(4),
            hide_index=True
        )

        # ----------------------------------------------------
        # METRIC COMPARISON
        # ----------------------------------------------------

        st.subheader("Metric comparison")

        metric_data = summary.melt(

            id_vars="algorithm",

            value_vars=[
                "dice",
                "iou",
                "precision",
                "recall"
            ],

            var_name="metric",

            value_name="score"
        )

        figure = px.bar(

            metric_data,

            x="metric",

            y="score",

            color="algorithm",

            barmode="group",

            color_discrete_map=self._colors,

            text_auto=".3f",

            labels={
                "metric": "Evaluation metric",
                "score": "Score"
            }
        )

        figure.update_yaxes(
            range=[0, 1.05]
        )

        st.plotly_chart(figure)

        # ----------------------------------------------------
        # PERFORMANCE DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "Performance consistency"
        )

        selected_metric = st.selectbox(
            "Select evaluation metric",
            [
                "dice",
                "iou",
                "precision",
                "recall"
            ]
        )

        distribution = px.box(

            results,

            x="algorithm",

            y=selected_metric,

            color="algorithm",

            color_discrete_map=self._colors,

            points="outliers"
        )

        distribution.update_yaxes(
            range=[0, 1.05]
        )

        st.plotly_chart(
            distribution
        )

        histogram = px.histogram(

            results,

            x=selected_metric,

            color="algorithm",

            barmode="overlay",

            opacity=0.65,

            nbins=30,

            color_discrete_map=self._colors
        )

        st.plotly_chart(histogram)

        # ----------------------------------------------------
        # INFERENCE PERFORMANCE
        # ----------------------------------------------------

        st.subheader("Inference time")

        timing = px.bar(

            summary,

            x="algorithm",

            y="inference_ms",

            color="algorithm",

            color_discrete_map=self._colors,

            text_auto=".2f",

            labels={

                "inference_ms":
                    "Mean inference time (ms)"
            }
        )

        st.plotly_chart(timing)

        st.caption(
            "The original benchmark used U-Net "
            "on the Apple GPU and GrabCut on the CPU. "
            "Inference times can vary between devices "
            "and benchmark runs."
        )

        # ----------------------------------------------------
        # CSV DOWNLOADS
        # ----------------------------------------------------

        st.download_button(

            "Download comparison CSV",

            summary.to_csv(index=False),

            file_name="comparison_summary.csv",

            mime="text/csv"
        )

        st.download_button(

            "Download per-image results",

            results.to_csv(index=False),

            file_name="per_image_results.csv",

            mime="text/csv"
        )

    # ========================================================
    # TRAINING ANALYTICS PAGE
    # ========================================================

    def _training(self):

        st.title("U-Net Training Analytics")

        history = self._load_csv(
            self._history_path
        )

        if history is None:
            return

        required_columns = {
            "epoch",
            "val_dice",
            "val_iou"
        }

        if not required_columns.issubset(
            history.columns
        ):

            st.error(
                "Training history is incomplete."
            )

            return

        if history["val_dice"].dropna().empty:

            st.error(
                "No valid validation Dice scores "
                "were found."
            )

            return

        best = history.loc[
            history["val_dice"].idxmax()
        ]

        cols = st.columns(4)

        cols[0].metric(
            "Completed epochs",
            len(history)
        )

        cols[1].metric(
            "Best validation Dice",
            f"{best['val_dice']:.2%}"
        )

        cols[2].metric(
            "Best checkpoint epoch",
            int(best["epoch"])
        )

        cols[3].metric(
            "Validation IoU at best epoch",
            f"{best['val_iou']:.2%}"
        )

        st.divider()

        charts = [

            (
                "Training and validation loss",
                [
                    "train_loss",
                    "val_loss"
                ]
            ),

            (
                "Dice coefficient",
                [
                    "train_dice",
                    "val_dice"
                ]
            ),

            (
                "Intersection over Union",
                [
                    "train_iou",
                    "val_iou"
                ]
            ),

            (
                "Precision",
                [
                    "train_precision",
                    "val_precision"
                ]
            ),

            (
                "Recall",
                [
                    "train_recall",
                    "val_recall"
                ]
            ),

            (
                "Learning rate schedule",
                ["learning_rate"]
            )
        ]

        for title, columns in charts:

            st.subheader(title)

            if not set(columns).issubset(
                history.columns
            ):

                st.warning(
                    f"Data unavailable for {title}."
                )

                continue

            figure = px.line(

                history,

                x="epoch",

                y=columns,

                markers=True,

                labels={
                    "epoch": "Epoch",
                    "value": "Value",
                    "variable": "Metric"
                }
            )

            st.plotly_chart(figure)

        with st.expander(
            "Complete training history"
        ):

            st.dataframe(
                history.round(6),
                hide_index=True
            )

        st.download_button(

            "Download Training History",

            history.to_csv(index=False),

            file_name="training_history.csv",

            mime="text/csv"
        )

    # ========================================================
    # ABOUT PROJECT PAGE
    # ========================================================

    def _about(self):

        st.title("About SkinSegAI")

        st.markdown("""
        ### Project objective

        Develop an object-oriented application for
        skin lesion segmentation using a custom U-Net
        neural network and compare it with GrabCut.

        ### Dataset

        ISIC 2018 Task 1, containing 2,594
        dermoscopic images and corresponding
        segmentation masks.

        ### Dataset preparation

        - Image and mask validation
        - Metadata-based group-aware splitting
        - Exact duplicate image detection
        - Reproducible dataset partitions
        - Image resizing to 256 x 256
        - Training-only data augmentation
        - Independent test-set evaluation

        ### Deep learning architecture

        - Custom lightweight U-Net
        - Four encoder and decoder levels
        - Skip connections
        - 1,942,577 trainable parameters
        - Combined BCE and Dice Loss
        - AdamW optimizer
        - Learning-rate scheduling
        - Early stopping and checkpointing

        ### Traditional segmentation

        GrabCut uses Gaussian Mixture Models and
        graph-based optimization without requiring
        neural network training.

        ### Object-oriented programming

        **Abstraction:** BaseSegmenter defines
        a common segmentation interface.

        **Inheritance:** UNetSegmenter and
        GrabCutSegmenter inherit from BaseSegmenter.

        **Polymorphism:** Both algorithms implement
        the same segment() method using different
        internal implementations.

        **Encapsulation:** Model weights, thresholds
        and processing logic are managed internally.

        ### Design patterns and principles

        - Strategy Pattern
        - Facade Pattern
        - Dependency Injection
        - Composition
        - Single Responsibility Principle
        - Open-Closed Principle
        - Separation of Concerns

        ### Lesion measurements

        - Area in square pixels
        - Percentage image coverage
        - Approximate perimeter
        - Circularity
        - Reflection-based asymmetry
        - Bounding box
        - Centroid coordinates

        ### Evaluation metrics

        - Dice coefficient
        - Intersection over Union
        - Precision
        - Recall
        - Inference time
        - Performance standard deviation

        ### Limitations

        - The model was trained on dermoscopic images.
        - Missing patient identifiers prevent a
          guarantee of complete patient independence.
        - Resizing can distort image geometry.
        - GrabCut may struggle with border-touching
          lesions.
        - Geometrical measurements are approximate.
        - Asymmetry depends on image orientation.
        - Physical dimensions require calibration.
        - Inference performance depends on hardware.
        - Segmentation is not cancer classification.
        - The application is not clinically validated.

        ### Technologies

        Python, PyTorch, OpenCV, NumPy, Pandas,
        Pillow, Plotly and Streamlit.
        """)

        st.warning(
            "This application is intended for academic "
            "and research demonstrations only. "
            "It is not a medical diagnostic tool."
        )

    # ========================================================
    # APPLICATION NAVIGATION
    # ========================================================

    def run(self):

        st.sidebar.title("SkinSegAI")

        st.sidebar.caption(
            "Skin Lesion Segmentation System"
        )

        pages = {

            "Dashboard":
                self._dashboard,

            "Live Segmentation":
                self._segmentation,

            "Model Comparison":
                self._comparison,

            "Training Analytics":
                self._training,

            "About Project":
                self._about
        }

        selection = st.sidebar.radio(

            "Navigation",

            list(pages.keys())
        )

        st.sidebar.divider()

        st.sidebar.caption(
            "U-Net | GrabCut | ISIC 2018"
        )

        st.sidebar.caption(
            "Academic research application"
        )

        pages[selection]()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    st.set_page_config(

        page_title="SkinSegAI",

        page_icon="🔬",

        layout="wide",

        initial_sidebar_state="expanded"
    )

    app = SkinSegApp()

    app.run()


if __name__ == "__main__":

    main()
