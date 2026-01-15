# Chest X-ray Classification using CNN

This project is a Convolutional Neural Network (CNN)–based system built to classify chest X-ray images into three categories:

- **COVID-19**
- **Normal**
- **Pneumonia**

The main goal of this project was to build an accurate and well-balanced medical image classifier **using a custom CNN**, while keeping training efficient on a **CPU-only setup**.

After multiple rounds of tuning, the final model achieves **~95.5% test accuracy** with a **balanced confusion matrix** across all three classes.

📄 **Project Report:** 
https://drive.google.com/file/d/1iUI-GHdQI4Hr5oYi3GQgyg0UGR3ffXc_/view?usp=sharing


---

## Why this project?

Chest X-ray analysis is an important problem in medical imaging, especially for respiratory diseases where visual patterns often overlap.  
This project focuses not just on accuracy, but also on **class balance and generalization**, which are critical in real-world medical applications.

---

## Key Features

- Custom CNN architecture (built from scratch)
- Offline balanced data augmentation
- Class-weighted loss to handle imbalance
- Label smoothing to improve generalization
- Early stopping and learning-rate scheduling
- Confusion matrix and performance visualization
- Single-image prediction support
- Optimized for CPU-based training

---

## Model Overview

- **Input size:** 48 × 48 RGB chest X-ray images  
- **Architecture:** Convolution → Pooling → Dense layers  
- **Activation:** ReLU (hidden layers), Softmax (output)  
- **Loss function:** Categorical Cross-Entropy with label smoothing  

The model was iteratively tuned to reduce overfitting and improve separation between visually similar classes such as Normal and Pneumonia.

---

## Results

- **Test Accuracy:** ~95.5%
- **Balanced recall** across COVID, Normal, and Pneumonia classes
- **Training time:** ~6 minutes on CPU
- Stable training and validation curves (no overfitting)

---

## Future Improvements

- Training with higher-resolution images
- Transfer learning using pretrained models
- Grad-CAM visualizations for model interpretability
- Deployment as a web or desktop application

---
