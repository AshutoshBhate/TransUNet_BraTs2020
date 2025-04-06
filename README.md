# TransUNet for Brain Tumor Segmentation on BraTS 2020

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![TransUNet Segmentation Example](images/segmentation_example.png)  <!-- Add example images if available -->

State-of-the-art brain tumor segmentation using a hybrid Transformer-CNN architecture, achieving **86.95% mean Dice score** on BraTS 2020 validation data.

## Key Features

- **Hybrid Architecture**: Combines EfficientNet-B4 CNN backbone with Vision Transformer (ViT)
- **Advanced Training**:
  - Mixed-precision training (FP16/FP32)
  - Gradient accumulation & clipping
  - Composite loss (Tversky-Focal + Boundary Loss)
- **Medical Imaging Focus**:
  - 4-channel MRI input (FLAIR, T1, T1CE, T2)
  - Class-balanced weight computation
  - Tumor border emphasis
- **Reproducible Results**: Detailed configuration system with hyperparameters
- **Visualization Tools**: Slice-by-slice predictions with modality comparison

## Performance Highlights

| Metric        | ET      | TC      | WT      | Mean    |
|---------------|---------|---------|---------|---------|
| Dice Score    | 0.8600  | 0.8935  | 0.8550  | 0.8695  |
| IoU           | 0.8110  | 0.8432  | 0.7985  | 0.8176  |

*Results on BraTS 2020 validation set (n=305 samples)*

## Quick Start

### 1. Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.7+
- BraTS 2020 Dataset ([Download](https://www.med.upenn.edu/cbica/brats2020/))

### 2. Installation

```bash
git clone https://github.com/yourusername/TransUNet_BraTs2020.git
cd TransUNet_BraTs2020

# Create and activate environment
conda create -n transunet python=3.8 -y
conda activate transunet

# Install dependencies
pip install -r requirements.txt