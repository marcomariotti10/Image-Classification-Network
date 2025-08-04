# Binary Image Classification with ConvNeXt and Transfer Learning

## 📘 Overview

This project implements a binary image classification pipeline using the **ConvNeXt Small** architecture with **transfer learning** from pre-trained ImageNet weights. The goal is to classify biological images as `healthy` or `unhealthy`.

The workflow includes:
- Data loading and preprocessing
- Stratified train/validation/test splitting
- Model training with early stopping and learning rate scheduling
- Fine-tuning the ConvNeXt model
- Evaluation on the test set with comprehensive metrics

---

## ✨ Features

- Load and inspect dataset from `.npz` files
- Class-balanced splitting using stratified sampling
- ConvNeXt-based deep learning model with:
  - Data augmentation
  - Frozen pre-trained backbone
  - Progressive fine-tuning
- Multi-phase training strategy
- Performance visualization and confusion matrix plotting
- Detailed evaluation metrics: accuracy, precision, recall, F1-score

---

## 📂 Dataset

The dataset is expected in a compressed NumPy format:

- `Balanced_data_final.npz`: contains two arrays:
  - `data`: images (shape: `n x 96 x 96 x 3`)
  - `labels`: text labels (`'healthy'` or `'unhealthy'`)

Class distribution is analyzed after loading.

---

## 📦 Requirements

To run this project, you need the following packages installed:

- Python 3.7+
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn
- seaborn
- matplotlib
- OpenCV (`cv2`)
