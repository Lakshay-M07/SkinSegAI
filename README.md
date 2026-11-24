# SkinSegAI

### Skin Lesion Segmentation and Comparative Analysis Using U-Net and GrabCut

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

## Overview

SkinSegAI is an object-oriented, GUI-based skin lesion segmentation and comparative analysis application developed using Python.

The system implements a custom lightweight U-Net neural network trained on dermoscopic images from the ISIC 2018 dataset and compares its performance against GrabCut, a traditional graph-based image segmentation algorithm.

The application integrates deep learning, image processing, quantitative performance evaluation, lesion morphology analysis and interactive visualization within a modular software architecture.

## Key Features

- Custom U-Net architecture trained from scratch.
- Automatic GrabCut segmentation.
- Side-by-side algorithm comparison.
- Interactive image segmentation interface.
- Adjustable prediction threshold and overlay opacity.
- Binary mask and lesion boundary visualization.
- Lesion morphology and geometrical analysis.
- Optional ground-truth mask evaluation.
- Interactive training and performance analytics.
- Downloadable segmentation masks and analytical reports.
- Metadata-based group-aware dataset splitting.
- Apple Silicon GPU acceleration with CPU fallback.
- Modular object-oriented architecture.

## Experimental Results

Both segmentation algorithms were evaluated on the same 390 held-out test images.

| Evaluation Metric | U-Net | GrabCut |
|---|---:|---:|
| Dice Coefficient | **0.8687** | 0.6146 |
| Intersection over Union | **0.7929** | 0.5120 |
| Precision | **0.8834** | 0.6021 |
| Recall | **0.8957** | 0.8697 |
| Dice Standard Deviation | **0.1550** | 0.3082 |
| Mean Inference Time | **19.84 ms** | 592.28 ms |
| Median Inference Time | **20.47 ms** | 500.33 ms |

### Key Findings

- U-Net achieved a mean test Dice coefficient of 86.87%.
- U-Net demonstrated a 25.41 percentage-point improvement in Dice over GrabCut.
- U-Net achieved approximately 29.9 times faster inference in the recorded benchmark.
- U-Net demonstrated greater segmentation consistency across the test dataset.
- The best recorded validation Dice coefficient was 85.77%.

The inference benchmark was conducted using Apple MPS acceleration for U-Net and CPU execution for GrabCut. Therefore, the speed comparison includes differences in hardware acceleration.

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

## U-Net Architecture

The custom U-Net consists of four encoder blocks, a bottleneck, four decoder blocks and skip connections.

| Parameter | Configuration |
|---|---|
| Architecture | Lightweight 2D U-Net |
| Input Channels | 3 |
| Output Channels | 1 |
| Input Resolution | 256 × 256 |
| Encoder Filters | 16, 32, 64, 128 |
| Bottleneck Filters | 256 |
| Trainable Parameters | 1,942,577 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy + Dice Loss |
| Prediction Threshold | 0.5 |
| Hardware Acceleration | Apple MPS / CPU |

The model incorporates learning-rate scheduling, early stopping and best-model checkpointing.

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
| UNet | Neural network architecture |
| UNetSegmenter | Deep learning inference |
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
| PyTorch | Deep learning and U-Net training |
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

Launch Streamlit:

```bash
python -m streamlit run gui/app.py
```

Open the application in your browser:

http://localhost:8501

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
python -m scripts.train_model --smoke
```

Train U-Net:

```bash
python -m scripts.train_model --epochs 30 --batch-size 4
```

Training automatically generates model checkpoints and epoch-wise performance history.

## Model Evaluation

Execute the comparative benchmark:

```bash
python -m scripts.evaluate_models
```

The evaluation generates:

- Per-image segmentation metrics.
- Aggregated algorithm comparison.
- Inference timing measurements.
- Lesion coverage statistics.

Results are stored in `artifacts/metrics/`.

## Verification and Testing

The project includes dedicated verification scripts for its major components.

```bash
python -m scripts.verify_dataset
python -m scripts.verify_model
python -m scripts.verify_segmenters
python -m scripts.verify_analysis
python -m scripts.verify_service
```

Verification covers dataset integrity, tensor dimensions, forward propagation, backpropagation, segmentation outputs, morphology calculations, backend integration and report generation.

## Future Enhancements

- U-Net pixel probability heatmaps.
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
