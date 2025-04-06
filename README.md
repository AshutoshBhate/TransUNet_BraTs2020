# TransUNet_BraTs2020

This repository contains the implementation of **TransUNet** for brain tumor segmentation using the BraTS 2020 dataset. TransUNet is a hybrid architecture combining convolutional neural networks (CNNs) and transformers, designed for medical image segmentation tasks.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
Brain tumor segmentation is a critical task in medical imaging, aiding in diagnosis and treatment planning. This project implements TransUNet, which leverages the strengths of CNNs for local feature extraction and transformers for global context modeling, to achieve state-of-the-art performance on the BraTS 2020 dataset.

## Features
- Hybrid architecture combining CNNs and transformers.
- Preprocessing pipeline for BraTS 2020 dataset.
- Support for multi-class segmentation (e.g., enhancing tumor, tumor core, whole tumor).
- Configurable training and evaluation scripts.
- Visualization of segmentation results.

## Dataset
The BraTS 2020 dataset contains multi-modal MRI scans (T1, T1Gd, T2, FLAIR) with ground truth annotations for brain tumor regions. You can download the dataset from the [BraTS 2020 Challenge website](https://www.med.upenn.edu/cbica/brats2020/).

### Preprocessing
Ensure the dataset is preprocessed to normalize intensities and resize images to the required input dimensions. Use the provided preprocessing scripts in the `data/` directory.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/TransUNet_BraTs2020.git
    cd TransUNet_BraTs2020
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage
### Preprocessing
Run the preprocessing script to prepare the dataset:
```bash
python preprocess.py --data_dir /path/to/brats2020 --output_dir /path/to/output
```

### Training
Train the model using the following command:
```bash
python train.py --config configs/train_config.yaml
```

### Evaluation
Evaluate the trained model on the validation set:
```bash
python evaluate.py --model_path /path/to/model.pth --data_dir /path/to/validation_data
```

### Visualization
Visualize the segmentation results:
```bash
python visualize.py --model_path /path/to/model.pth --data_dir /path/to/test_data
```

## Training
The training pipeline is configurable via YAML files in the `configs/` directory. Key parameters include:
- Learning rate
- Batch size
- Number of epochs
- Optimizer and loss function

Modify `configs/train_config.yaml` to customize training.

## Evaluation
The evaluation script computes metrics such as Dice coefficient, sensitivity, and specificity. Results are saved in the `results/` directory.

## Results
The model achieves competitive performance on the BraTS 2020 dataset. Example segmentation results and quantitative metrics are provided in the `results/` directory.

## Model Architecture
TransUNet combines:
- **CNN Encoder**: Extracts local features from input images.
- **Transformer Block**: Captures global context using self-attention mechanisms.
- **Decoder**: Reconstructs segmentation maps from encoded features.

Refer to the `models/` directory for implementation details.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a detailed description.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The BraTS 2020 organizers for providing the dataset.
- The authors of TransUNet for their groundbreaking work.

For questions or feedback, please open an issue or contact the repository maintainer.