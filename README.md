# SkinSegAI

### Skin Lesion Segmentation and Comparative Analysis Using U-Net and GrabCut

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

## Overview

SkinSegAI is an object-oriented, GUI-based skin lesion segmentation and comparative analysis application developed using Python.

The system supports both a custom lightweight U-Net and ImageNet-pretrained encoder U-Net backbones (`resnet34_unet`, `efficientnet_b2_unet`) trained on dermoscopic images from the ISIC 2018 dataset with a composite edge-sharpening loss, and compares performance against GrabCut, a traditional graph-based image segmentation algorithm.

The application integrates deep learning, image processing, pixel-wise uncertainty estimation, quantitative performance evaluation, lesion morphology analysis and interactive visualization within a modular software architecture.

## Key Features

- Selectable U-Net architectures: `resnet34_unet` (ImageNet-pretrained), `efficientnet_b2_unet` (ImageNet-pretrained), and `custom_unet` (from-scratch).
- Composite edge-sharpening loss (`CompositeSegmentationLoss`) combining Soft Dice, BCE with Logits, and a differentiable morphological Boundary Loss.
- Pixel-wise uncertainty heatmaps (binary predictive entropy + optional MC Dropout) and Border Uncertainty Score.
- Automatic GrabCut segmentation.
- Side-by-side algorithm comparison.
- Interactive image segmentation interface with `@st.cache_resource` model caching.
- Adjustable prediction threshold and overlay opacity.
- Binary mask and lesion boundary visualization.
- Lesion morphology and geometrical analysis.
- Optional ground-truth mask evaluation.
- Interactive training and performance analytics (including per-component loss curves).
- Downloadable segmentation masks and analytical reports.
- Metadata-based group-aware dataset splitting.
- NVIDIA CUDA and Apple Silicon MPS GPU acceleration with CPU fallback.
- Modular object-oriented architecture.

## Experimental Results

Both segmentation algorithms were evaluated on the same 390 held-out test images using the trained `resnet34_unet` checkpoint.

| Evaluation Metric | U-Net (`resnet34_unet`) | GrabCut |
|---|---:|---:|
| Dice Coefficient | **0.8779** | 0.6146 |
| Intersection over Union | **0.8023** | 0.5120 |
| Precision | **0.8893** | 0.6021 |
| Recall | **0.9014** | 0.8697 |
| Dice Standard Deviation | **0.1370** | 0.3082 |
| Mean Inference Time | **126.01 ms** | 1933.82 ms |
| Median Inference Time | **166.92 ms** | 1581.80 ms |

### Key Findings

- U-Net (`resnet34_unet`) achieved a mean test Dice coefficient of 87.79% and IoU of 80.23%.
- U-Net demonstrated a 26.32 percentage-point improvement in Dice over GrabCut.
- U-Net achieved approximately 15.3 times faster mean inference in the recorded benchmark.
- U-Net demonstrated greater segmentation consistency across the test dataset (Dice std `0.1370` vs. `0.3082`).
- The best recorded validation Dice coefficient was 87.65%.

The inference benchmark was conducted using NVIDIA RTX 3050 CUDA acceleration for U-Net and CPU execution for GrabCut. Therefore, the speed comparison includes differences in hardware acceleration.

These results represent segmentation performance, not clinical diagnostic accuracy.

## Dataset

The application uses the ISIC 2018 Challenge — Task 1: Lesion Segmentation dataset.

- Total images: 2,594
- Total ground-truth masks: 2,594
- Image type: Dermoscopic RGB images
- Model input resolution: 256 × 256 pixels
- Approximate dataset split: 70% training, 15% validation and 15% testing
- Test images: 390

Dataset partitioning incorporates available patient identifiers, lesion identifiers and exact image hashes to reduce data leakage.

Official dataset:

https://challenge.isic-archive.com/data/#2018

Training metadata:

https://api.isic-archive.com/collections/63/metadata/

The original dataset, processed images and patient metadata are not included in this repository.

## Model Architectures & Loss Function

The model factory (`src/models/model_factory.py`) supports three architectures with automatic ImageNet normalization for pretrained encoders and checkpoint-driven architecture detection during inference:

| Architecture Key | Description | Normalization |
|---|---|---|
| `resnet34_unet` (default) | `smp.Unet` with ImageNet-pretrained ResNet-34 encoder | ImageNet mean/std |
| `efficientnet_b2_unet` | `smp.Unet` with ImageNet-pretrained EfficientNet-B2 encoder | ImageNet mean/std |
| `custom_unet` | Lightweight 2D U-Net trained from scratch (1,942,577 params) | `[0, 1]` scaling |

| Parameter | Configuration |
|---|---|
| Input / Output Channels | 3 (RGB) / 1 (Logits) |
| Input Resolution | 256 × 256 |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Learning Rate | 0.001 (`ReduceLROnPlateau`) |
| Loss Function | `CompositeSegmentationLoss` (0.4 Dice + 0.3 BCE + 0.3 Boundary) |
| Prediction Threshold | 0.5 |
| Hardware Acceleration | NVIDIA CUDA / Apple MPS / CPU |

The model incorporates learning-rate scheduling, early stopping, per-component loss logging, and best-model checkpointing.

## Object-Oriented Programming

The project demonstrates the four fundamental pillars of object-oriented programming.

| OOP Principle | Implementation |
|---|---|
| Abstraction | Abstract BaseSegmenter interface |
| Encapsulation | Internal model weights, configurations and processing logic |
| Inheritance | UNetSegmenter and GrabCutSegmenter inherit from BaseSegmenter |
| Polymorphism | Both algorithms implement a common segment() method |

### Design Patterns and Software Engineering Principles

- Strategy Design Pattern
- Facade Design Pattern
- Dependency Injection
- Composition
- Single Responsibility Principle
- Open-Closed Principle
- Separation of Concerns

### Core Classes

| Class | Responsibility |
|---|---|
| DatasetManager | Dataset discovery and validation |
| MetadataLoader | Automatic metadata discovery |
| DatasetSplitter | Group-aware dataset partitioning |
| ImagePreprocessor | Image and mask preprocessing |
| LesionDataset | PyTorch dataset loading and augmentation |
| BaseSegmenter | Abstract segmentation interface |
| UNet / model_factory | Neural network architectures and pretrained encoder builder |
| CompositeSegmentationLoss | Combined Soft Dice, BCE with Logits, and morphological Boundary Loss |
| UNetSegmenter | Deep learning inference and predictive entropy / MC Dropout uncertainty |
| GrabCutSegmenter | Traditional segmentation |
| ModelTrainer | Training, validation and checkpointing |
| MetricsEvaluator | Quantitative performance evaluation |
| SegmentationBenchmark | Comparative benchmarking |
| LesionAnalyzer | Geometrical lesion measurements |
| SegmentationResult | Structured segmentation results |
| SegmentationService | Backend orchestration |
| ReportExporter | ZIP report generation |
| SkinSegApp | Streamlit graphical user interface |

## Application Workflow

1. Upload a dermoscopic image or select a validation dataset sample.
2. Validate and preprocess the input image.
3. Select U-Net, GrabCut or comparison mode.
4. Generate a binary segmentation mask.
5. Restore the mask to the original image dimensions.
6. Calculate lesion morphology measurements.
7. Evaluate predictions if a ground-truth mask is available.
8. Display interactive visualizations.
9. Export segmentation results and analytical reports.

## Lesion Morphology Analysis

The application calculates:

- Lesion area in square pixels.
- Percentage image coverage.
- Approximate perimeter.
- Circularity.
- Reflection-based asymmetry.
- Bounding box.
- Centroid coordinates.

All measurements are pixel-based estimates and should not be interpreted as clinical diagnostic indicators.

## Graphical User Interface

The application provides five interactive sections.

| Page | Description |
|---|---|
| Dashboard | Project overview and benchmark statistics |
| Live Segmentation | Interactive segmentation and lesion analysis |
| Model Comparison | Quantitative algorithm comparison |
| Training Analytics | Training loss and performance curves |
| About Project | Architecture, methodology and OOP concepts |

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| PyTorch | Deep learning, custom U-Net training, and CUDA/MPS acceleration |
| segmentation-models-pytorch & timm | ImageNet-pretrained U-Net encoder backbones |
| OpenCV | GrabCut and image processing |
| Streamlit | Graphical user interface |
| NumPy | Numerical computation |
| Pandas | Dataset and benchmark analysis |
| Pillow | Image loading and manipulation |
| Plotly | Interactive data visualization |
| Matplotlib | Scientific visualization |
| Scikit-learn | Dataset partitioning |
| Pytest | Automated testing framework |

## Project Structure

```
SkinSegAI/
├── .streamlit/
│   └── config.toml
├── artifacts/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── training_history.csv
│   └── metrics/
│       ├── comparison_summary.csv
│       └── per_image_results.csv
├── configs/
├── data/
│   ├── raw/
│   ├── processed_grouped/
│   └── splits_grouped/
├── docs/
├── gui/
│   └── app.py
├── scripts/
│   ├── verify_dataset.py
│   ├── prepare_dataset.py
│   ├── verify_model.py
│   ├── train_model.py
│   ├── verify_segmenters.py
│   ├── evaluate_models.py
│   ├── verify_analysis.py
│   └── verify_service.py
├── src/
│   ├── analysis/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── segmentation/
│   ├── services/
│   ├── training/
│   └── utils/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

Dataset directories are created locally and are excluded from version control where appropriate.

## Installation

Clone this repository using its GitHub clone URL and navigate to the SkinSegAI project directory.

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original development and GPU testing were performed on macOS with Apple Silicon. Dependency compatibility may vary across Python versions and operating systems.

## Running the Application

Ensure the trained checkpoint exists at:

```
artifacts/checkpoints/best_model.pth
```

### Option 1: React + FastAPI Workstation (Recommended)

1. Start the FastAPI backend (Terminal 1):

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

2. Start the React + Vite frontend (Terminal 2):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### Option 2: Streamlit Workstation

```bash
python -m streamlit run gui/app.py
```

Open http://localhost:8501 in your browser.

The original dataset is not required for inference on uploaded images when the trained checkpoint is available. The built-in validation dataset sample feature requires local processed dataset files.

## Dataset Setup

To reproduce training and evaluation, download the ISIC 2018 Task 1 training images, segmentation masks and metadata.

Organize them as follows:

```
data/raw/
├── ISIC2018_Task1-2_Training_Input/
├── ISIC2018_Task1_Training_GroundTruth/
└── metadata.csv
```

Dated metadata CSV filenames are also supported through automatic metadata discovery.

Verify the dataset:

```bash
python -m scripts.verify_dataset
```

Prepare the dataset:

```bash
python -m scripts.prepare_dataset
```

## Model Training

Verify the model architecture:

```bash
python -m scripts.verify_model
```

Run a training smoke test:

```bash
python scripts/train_model.py --arch resnet34_unet --smoke
```

Train U-Net (default `resnet34_unet` with `CompositeSegmentationLoss`):

```bash
python scripts/train_model.py --arch resnet34_unet --epochs 20 --batch-size 8
```

Available training CLI flags:

- `--arch`: `resnet34_unet` (default), `efficientnet_b2_unet`, or `custom_unet`
- `--epochs`: number of training epochs (default `30`)
- `--batch-size`: training/validation batch size (default `4`)
- `--dice-weight`, `--bce-weight`, `--boundary-weight`: `CompositeSegmentationLoss` weights (defaults `0.4`, `0.3`, `0.3`)
- `--smoke`: run a 1-epoch smoke test on a small subset

Training automatically generates model checkpoints (with `arch` and `loss_weights` metadata) and epoch-wise component loss/performance history.

## Model Evaluation

Execute the comparative benchmark:

```bash
python scripts/evaluate_models.py
```

The evaluation generates:

- Per-image segmentation metrics.
- Aggregated algorithm comparison.
- Inference timing measurements.
- Lesion coverage statistics.

Results are stored in `artifacts/metrics/`.

## Verification and Testing

The project includes automated `pytest` unit tests and dedicated verification scripts for its major components.

```bash
python -m pytest tests/ -v
python -m scripts.verify_dataset
python -m scripts.verify_model
python -m scripts.verify_segmenters
python -m scripts.verify_analysis
python -m scripts.verify_service
```

Tests and verification cover model architecture output shapes, `CompositeSegmentationLoss` gradients and edge-sharpening behavior, predictive entropy and MC Dropout uncertainty, dataset integrity, morphology calculations, backend integration, and report generation.

## Future Enhancements

- Segmentation error visualization.
- Algorithm disagreement maps.
- Individual connected-region analysis.
- Batch image segmentation.
- Advanced statistical performance analysis.
- PDF report generation.
- Further GUI and accessibility improvements.
- Cross-platform compatibility testing.

## Limitations

- The model was trained on dermoscopic images.
- Performance on ordinary smartphone images is not established.
- Missing patient identifiers prevent a guarantee of complete patient-level independence.
- Image resizing may introduce geometric distortion.
- GrabCut may struggle with lesions touching image boundaries.
- U-Net may overlook smaller secondary pigmented regions.
- Morphological measurements are approximate.
- Physical measurements require spatial calibration.
- The application does not classify lesions as benign or malignant.

## Disclaimer

SkinSegAI is an academic and educational research project.

The application is not clinically validated and must not be used as a substitute for professional medical diagnosis, treatment or dermatological examination.

## Acknowledgements

- International Skin Imaging Collaboration (ISIC)
- U-Net: Convolutional Networks for Biomedical Image Segmentation — Ronneberger, Fischer and Brox
- GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts — Rother, Kolmogorov and Blake
- PyTorch, OpenCV and Streamlit open-source communities
