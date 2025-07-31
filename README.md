# Age Detection with Fine-Tuned CNN

This repository contains the implementation of an age detection system using a fine-tuned pre-trained Convolutional Neural Network (CNN) model. The project leverages the UTKFace dataset for training and evaluation.

## Table of Contents

- [Features](#features)
- [Expected Performance](#expected-performance)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pre-trained CNN Models**: Utilizes popular architectures such as ResNet50, ResNet34, and EfficientNet-B0 for robust feature extraction.
- **Two-Phase Fine-Tuning Strategy**: Employs a structured fine-tuning approach to optimize model performance.
- **Comprehensive Evaluation Metrics**: Includes Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R2 Score, and age group accuracies (within ±5 and ±10 years).
- **Real-time Training Visualization**: Provides insights into the training process through loss and MAE curves.

## Expected Performance

Based on the provided notebook, the expected performance metrics are:

- **MAE**: 4-6 years
- **Accuracy (±5 years)**: 65-75%
- **Accuracy (±10 years)**: 85-90%

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/age-detection.git
   cd age-detection
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the **UTKFace Dataset**. This dataset consists of over 20,000 face images with annotations for age, gender, and ethnicity. The filename format is `[age]_[gender]_[race]_[date&time].jpg`.

**Download and Preparation:**

1. Download the UTKFace dataset (e.g., from Kaggle or its official source).
2. Place the dataset in a directory named `UTKFace_data` (or modify the `data_dir` variable in the notebook).
3. If the dataset is a zip file, ensure it is extracted into the `UTKFace_data` directory such that the images are directly accessible (e.g., `UTKFace_data/UTKFace/`).

## Model Architecture

The `AgeDetectionModel` is built upon pre-trained CNN backbones. It consists of:

- **Backbone**: A pre-trained model (ResNet50, ResNet34, or EfficientNet-B0) used for feature extraction. The final classification layer of the backbone is replaced with an `nn.Identity()` layer.
- **Age Regressor Head**: A sequential neural network comprising `Dropout`, `Linear`, and `ReLU` layers, designed to predict the age from the features extracted by the backbone. Weights are initialized using Xavier uniform initialization.

### Loss Function

The model can be trained with either:

- **Mean Squared Error (MSE) Loss**: Standard regression loss.
- **Focal MSE Loss**: A custom loss function designed to focus on harder examples by weighting the MSE loss based on prediction error. This can be enabled by setting `USE_FOCAL_LOSS = True` in the training configuration.

## Training

The training process involves a two-phase fine-tuning strategy:

1. **Phase 1: Training with Frozen Backbone**: The backbone parameters are frozen, and only the age regressor head is trained for a few epochs. This helps in quickly adapting the new head to the extracted features.
2. **Phase 2: End-to-End Fine-Tuning**: The entire model (backbone and regressor head) is unfrozen and fine-tuned with a lower learning rate. This allows the model to learn more specific features relevant to age detection.

### Configuration Parameters

Key training parameters include:

- `DEVICE`: `cuda` if GPU is available, otherwise `cpu`.
- `MODEL_NAME`: `resnet50`, `resnet34`, or `efficientnet_b0`.
- `EPOCHS`: Total number of training epochs.
- `LEARNING_RATE`: Initial learning rate for the optimizer.
- `USE_FOCAL_LOSS`: Boolean flag to enable Focal MSE Loss.
- `BATCH_SIZE`: Number of samples per batch.

An `AdamW` optimizer with a `ReduceLROnPlateau` scheduler is used to adjust the learning rate during training.

## Evaluation

Model performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual ages.
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual ages.
- **Root Mean Squared Error (RMSE)**: Square root of MSE.
- **R2 Score**: Coefficient of determination.
- **Accuracy (within ±5 years)**: Percentage of predictions within 5 years of the actual age.
- **Accuracy (within ±10 years)**: Percentage of predictions within 10 years of the actual age.

Training and validation loss and MAE curves are plotted to visualize the training progress.

## Usage

To run the age detection model:

1. Ensure you have followed the [Installation](#installation) and [Dataset](#dataset) steps.
2. Open the `agedetection.ipynb` Jupyter notebook.
3. Run all cells in the notebook. The notebook will guide you through data loading, model setup, training, and evaluation.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: A LICENSE file is not provided in the original notebook, this is a placeholder.)


