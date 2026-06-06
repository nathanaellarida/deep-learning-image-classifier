# Deep Learning Image Classifier

A convolutional neural network (CNN) image classifier built with TensorFlow and Keras, designed to run on Google Colab with images stored in Google Drive.

## Overview

This project trains a multi-class image classifier using a custom CNN architecture. The model dynamically detects class names from the training directory, making it flexible for any set of image categories.

## Architecture

- **Input**: 128×128 RGB images
- **Feature Extraction**: 4× Conv2D + MaxPooling2D blocks (32 → 64 → 128 → 256 filters)
- **Classifier Head**: Dense(512, ReLU) → Dropout(0.8) → Dense(N classes, Softmax)
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam (lr=0.0001)
- **Regularization**: EarlyStopping (patience=5)

## Requirements

```
tensorflow
keras
opencv-python (cv2)
numpy
matplotlib
```

## Usage

1. Upload the notebook to [Google Colab](https://colab.research.google.com/).
2. Mount your Google Drive and place your dataset in the following structure:

```
MyDrive/
└── <project-folder>/
    └── train/
        ├── class_1/
        │   ├── img1.jpg
        │   └── ...
        ├── class_2/
        │   └── ...
        └── class_N/
            └── ...
```

3. Update the `os.chdir(...)` path in the notebook to point to your folder.
4. Run all cells to train the model. The trained model is saved as `animals-cnn.h5`.
5. To test, set `TEST_IMG` to the path of a test image and run the final cell.

## Results

Training uses an 80/20 train-test split with early stopping to prevent overfitting. After training, predictions are displayed with the predicted class label and confidence score.
