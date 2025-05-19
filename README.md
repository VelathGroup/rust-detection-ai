
# Steel Surface Inspection Robot



## Overview

This repository contains a deep learning-based perception system for an autonomous steel surface inspection robot. The system detects steel surfaces, analyzes rust levels, identifies edges, and recommends appropriate sandblasting treatment parameters. The project implements a multi-model pipeline architecture that's designed for real-time processing on robotic platforms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.0+-green.svg)](https://github.com/ultralytics/ultralytics)

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Evaluation](#-evaluation)
- [Model Zoo](#-model-zoo)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

- **Multi-class object detection**: Identifies steel surfaces and rust presence using YOLOv8
- **Edge detection**: Generates precise edge maps for robot path planning
- **Rust level classification**: Categorizes rust levels according to ISO 8501-1 standards (SA2.5/SA3.0)
- **Automated decision-making**: Provides recommendations for blasting intensity, pressure, and distance
- **Visualization tools**: Displays detection results, edge maps, and treatment recommendations
- **Extensible pipeline**: Modular architecture makes it easy to add or replace components

## 🏗️ System Architecture

The pipeline consists of three main components:

1. **YOLO Steel/Rust Detector**: Identifies steel surfaces and presence of rust using YOLOv8
2. **UNet Edge Detector**: Generates edge maps for surface analysis and robot path planning
3. **MobileNetV3 Rust Classifier**: Categorizes rust levels to determine treatment intensity

![Pasted image](https://github.com/user-attachments/assets/722f5714-15cd-4c06-9c47-5a034afd80b4)

YOLO V8

![Pasted image (4)](https://github.com/user-attachments/assets/79ff6c18-a215-4126-96e0-c94c24342002)


  MobileNetV3

  ![Pasted image (5)](https://github.com/user-attachments/assets/c2d0e5f6-51b5-4c4c-a7be-b7f139238c43)\
  UNet






## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git LFS (for model files)

### Option 1: Quick Start with Pip

```bash
# Clone this repository
git clone https://github.com/your-username/steel-surface-inspection-robot.git
cd steel-surface-inspection-robot

# Install dependencies
pip install -r requirements.txt

# Set up project structure and download placeholder models
python setup.py
```

### Option 2: Install as a Package

```bash
# Clone this repository
git clone https://github.com/your-username/steel-surface-inspection-robot.git
cd steel-surface-inspection-robot

# Install the package
pip install -e .
```

### Option 3: Docker

```bash
# Build the Docker image
docker build -t steel-inspection-robot .

# Run the container
docker run -it --gpus all -v $(pwd)/data:/app/data steel-inspection-robot
```

## 🎮 Usage

### Running Inference

```bash
# Using a single image
python src/inference.py --image sample_data/sample_1.jpg

# Specifying model paths manually
python src/inference.py --image sample_data/sample_1.jpg \
                        --yolo_model models/yolo_best.pt \
                        --unet_model models/unet_edge_detector.pth \
                        --classifier_model models/rust_level_classifier.pth
```

### Example Results

<div align="center">
  <img src="docs/images/results_example.png" alt="Example Results" width="800"/>
</div>

### Using as a Python Module

```python
from src.inference import load_models, robot_perception_inference

# Load models
yolo_model, unet_model, rust_classifier, device = load_models()

# Run inference
results = robot_perception_inference(
    "sample_data/sample_1.jpg",
    yolo_model,
    unet_model,
    rust_classifier,
    device
)

# Access results
print(f"Steel detected: {results['steel_detected']}")
print(f"Rust level: {results['rust_level']}")
print(f"Recommended treatment: {results['decision']}")
```

## 📊 Data Preparation

### Dataset Structure

The project uses three datasets for training the different components:

1. **YOLO Dataset**: Steel and rust detection dataset in YOLO format
2. **Edge Dataset**: Images and edge masks for UNet training
3. **Rust Level Dataset**: Classified rust images by severity level

```
data/
├── yolo_dataset/
│   ├── images/
│   │   ├── train/
│   │   │   └── *.jpg
│   │   └── val/
│   │       └── *.jpg
│   ├── labels/
│   │   ├── train/
│   │   │   └── *.txt
│   │   └── val/
│   │       └── *.txt
│   └── data.yaml
├── edge_dataset/
│   ├── images/
│   │   └── *.jpg
│   └── masks/
│       └── *.png
└── rust_level_dataset/
    ├── train/
    │   ├── SA2.5/
    │   │   └── *.jpg
    │   └── SA3.0/
    │       └── *.jpg
    └── val/
        ├── SA2.5/
        │   └── *.jpg
        └── SA3.0/
            └── *.jpg
```

### Data Preparation Scripts

The project includes scripts to help prepare your data:

```bash
# Generate project folder structure
python folder_generator.py

# Prepare sample data (for testing)
python setup.py --force_download
```

## 🧠 Model Training

### Training the YOLO Steel/Rust Detector

```bash
python src/train_yolo.py --data_dir data/yolo_dataset \
                         --epochs 100 \
                         --batch_size 8 \
                         --img_size 416
```

### Training the UNet Edge Detector

```bash
python src/train_unet.py --data_dir data/edge_dataset \
                         --epochs 50 \
                         --batch_size 4
```

### Training the Rust Level Classifier

```bash
python src/train_classifier.py --data_dir data/rust_level_dataset \
                              --epochs 30 \
                              --batch_size 8 \
                              --model_type mobilenet_v3_small
```

## 📈 Evaluation

| Model | Accuracy | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------|-----------|--------|----------|---------|--------------|
| YOLO Steel/Rust Detector | % | % | % | % | % | % |
| UNet Edge Detector | % | - | - | - | - | - |
| Rust Level Classifier | % | % | % | % | - | - |

## 📦 Model Zoo

Pre-trained models are available for download:

| Model | Description | Size | Link |
|-------|-------------|------|------|
| YOLO Steel/Rust Detector | YOLOv8n trained on our steel/rust dataset | 15 MB | [Download](https://example.com/yolo_steel_rust_detector.pt) |
| UNet Edge Detector | Simple UNet for edge detection | 5 MB | [Download](https://example.com/unet_edge_detector.pth) |
| Rust Level Classifier | MobileNetV3 classifier for SA standards | 10 MB | [Download](https://example.com/rust_level_classifier.pth) |

## 📁 Project Structure

```
steel-surface-inspection-robot/
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
├── setup.py                   # Project setup utilities
├── folder_generator.py        # Creates folder structure
├── download_models.py         # Downloads pretrained models
├── models/                    # Model storage
├── data/                      # Dataset folders
│   ├── yolo_dataset/          # YOLO dataset
│   ├── edge_dataset/          # Edge detection dataset
│   └── rust_level_dataset/    # Rust classification dataset
├── results/                   # Output directory
├── sample_data/               # Sample test images
├── docs/                      # Documentation
│   └── images/                # Documentation images
└── src/                       # Source code
    ├── __init__.py            # Package initialization
    ├── inference.py           # Main inference pipeline
    ├── train_yolo.py          # YOLO training script
    ├── train_unet.py          # UNet training script
    ├── train_classifier.py    # Classifier training script
    └── utils.py               # Utility functions
```

## 📋 Requirements

```
torch>=1.10.0
torchvision>=0.11.0
ultralytics>=8.0.0
opencv-python>=4.6.0
pillow>=7.1.2
numpy>=1.23.0
matplotlib>=3.3.0
albumentations>=1.0.0
tqdm>=4.64.0
pyyaml>=5.3.1
scikit-learn>=1.0.0
```

## 🛣️ Roadmap

- [ ] Add support for video streams
- [ ] Improve real-time performance with TensorRT optimizations
- [ ] Add ROS integration for robot deployment
- [ ] Implement active learning pipeline for continuous improvement
- [ ] Add multi-view fusion for complete surface coverage
- [ ] Create web interface for remote monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

