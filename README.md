# Tuberculosis Detection from Chest X-rays using Deep Learning

## Project Overview

This project implements a deep learning model to detect tuberculosis (TB) from chest X-ray images. It uses a Convolutional Neural Network (CNN) trained on a dataset of chest X-rays, including both normal and TB-positive cases, along with their corresponding lung masks and clinical readings.

## Features

- Data preprocessing and loading from multiple directories
- CNN model architecture for image classification
- Training and evaluation of the model
- Prediction on new X-ray images
- Visualization of sample X-rays with predictions

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. 2. Install the required packages

## Usage

1. Prepare your dataset according to the structure .

2. Run the main script
3. The script will:
- Load and preprocess the data
- Train the CNN model
- Evaluate the model on the test set
- Save the trained model as 'trained_model.h5'
- Make a prediction on a sample X-ray image

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- 2 Dense layers (including the output layer with sigmoid activation)

## Performance

The model's performance is evaluated using binary cross-entropy loss and accuracy. After training, the test accuracy is printed.

## Sample Prediction

The script includes a sample prediction on a single X-ray image. It will display the image and print whether the X-ray indicates TB or not, along with a confidence score.

## Customization

You can modify the following aspects of the project:
- Model architecture in the `model = tf.keras.Sequential(...)` section
- Training parameters such as `epochs` and `batch_size`
- Input image dimensions (currently set to 224x224)
- Dataset split ratio (currently 80% train, 20% test)

## Contributing

Contributions to improve the model or extend its functionality are welcome. Please feel free to submit a pull request or open an issue.


## Acknowledgments

- Dataset source: [provide the source of your dataset here]
- This project is for educational purposes and should not be used for actual medical diagnosis without proper validation and approval from medical professionals.
