
# SkinSegAI — Backend Integration Guide

## 1. Supported Models

| Architecture ID | Display Name | Trained |
|---|---|---|
| custom_unet | Custom U-Net | Yes |
| resnet34_unet | ResNet-34 U-Net | Yes |
| efficientnet_b2_unet | EfficientNet-B2 U-Net | No |
| N/A | GrabCut | Training not required |

EfficientNet-B2 is supported by the model factory but does not
currently have a trained checkpoint.

## 2. Initialize the Segmentation Service

```python
from src.services.segmentation_service import SegmentationService

service = SegmentationService.from_architectures()
```

This loads Custom U-Net, ResNet-34 U-Net and GrabCut.

Hardware selection is automatic:
CUDA, Apple MPS, or CPU.

## 3. Retrieve Available Algorithms

```python
algorithms = service.available_algorithms
```

Expected result:

```python
[
    "Custom U-Net",
    "ResNet-34 U-Net",
    "GrabCut",
]
```

Use these names directly in frontend dropdown menus.

## 4. Initialize Selected Models

```python
service = SegmentationService.from_architectures(
    architectures=("resnet34_unet",),
    threshold=0.5,
    include_grabcut=True,
)
```

## 5. Perform Single-Model Segmentation

```python
result = service.segment(
    image=image,
    algorithm="ResNet-34 U-Net",
    alpha=0.35,
)
```

The image must be a PIL Image.

## 6. Compare All Registered Models

```python
results = service.compare(
    image=image,
    alpha=0.35,
)
```

Returns a list of SegmentationResult objects.

## 7. SegmentationResult Fields

| Field | Description |
|---|---|
| algorithm | Registered algorithm name |
| mask | Boolean NumPy segmentation mask |
| overlay | PIL segmentation overlay |
| boundary | PIL boundary visualization |
| measurements | Morphological measurements |
| inference_ms | Segmentation execution time |
| metrics | Ground-truth evaluation metrics |
| prob_map | Foreground probability map |
| uncertainty_map | Predictive entropy map |
| border_uncertainty | Mean boundary uncertainty |

Probability and uncertainty maps are unavailable for GrabCut.

Ground-truth metrics are None when no ground truth is supplied.

Border uncertainty is None for empty or completely filled masks.

All prediction maps match the original image dimensions.

## 8. Ground-Truth Evaluation

```python
result = service.segment(
    image=image,
    algorithm="ResNet-34 U-Net",
    ground_truth=ground_truth,
)
```

Ground truth must:

- Match the original image dimensions.
- Contain only binary values: 0 and 1.

Available metrics include Dice, IoU, precision and recall.

## 9. Model Artifact Registry

```python
from src.models.artifacts import ModelArtifactRegistry

registry = ModelArtifactRegistry()

trained_models = registry.trained_models()

artifacts = registry.get("resnet34_unet")
```

Available artifact properties:

- checkpoint
- training_history
- comparison_summary
- per_image_results
- is_trained
- has_training_history
- has_benchmark

Use the registry rather than hardcoding model paths.

## 10. Report Export

```python
from src.services.report_exporter import ReportExporter

exporter = ReportExporter()

report = exporter.export_zip(results)
```

The returned ZIP bytes can be passed to Streamlit's
download_button.

Each algorithm has its own report directory.

## 11. Verified Benchmark Results

Evaluated on 390 held-out ISIC 2018 images.

| Metric | GrabCut | Custom U-Net | ResNet-34 |
|---|---:|---:|---:|
| Dice | 0.6146 | 0.8687 | 0.8779 |
| IoU | 0.5120 | 0.7929 | 0.8023 |
| Precision | 0.6021 | 0.8834 | 0.8893 |
| Recall | 0.8697 | 0.8957 | 0.9014 |
| Dice SD | 0.3082 | 0.1550 | 0.1370 |
| Mean inference (ms) | 573.53 | 20.08 | 30.03 |

Neural networks were evaluated using Apple MPS.
GrabCut was evaluated using CPU.

## 12. Training and Evaluation

Training:

```bash
python -m scripts.train_model --arch custom_unet
python -m scripts.train_model --arch resnet34_unet
```

Evaluation:

```bash
python -m scripts.evaluate_models --arch custom_unet
python -m scripts.evaluate_models --arch resnet34_unet
```

New experiments are stored in unique directories under:

artifacts/runs/training/
artifacts/runs/evaluation/

Existing production checkpoints are not automatically replaced.

## 13. Frontend Integration Requirements

- Use the segmentation service for inference.
- Use the artifact registry for model discovery.
- Do not duplicate metric calculations.
- Do not hardcode architecture-specific checkpoint paths.
- Do not display unavailable models as trained.
- Handle None values for optional result fields.
- Do not treat predictive entropy as clinical confidence.
- Avoid sharing mutable model state across concurrent sessions.
- Keep the research and educational disclaimer visible.

## 14. Known Limitations

- EfficientNet-B2 has not been trained.
- MC Dropout is stochastic only when supported dropout
  layers are present.
- Concurrent inference safety has not been fully verified.
- The ZIP exporter does not yet export probability and
  uncertainty maps.
- Training smoke tests verify execution, not convergence.
- Missing patient and lesion IDs prevent guaranteed
  patient-level independence across the entire dataset.
- The application is intended for educational and
  research purposes, not clinical diagnosis.

## 15. Integration Status

Backend runtime regression tests completed successfully.

The frontend remains under separate development.
Frontend integration and end-to-end GUI verification
are still required.
