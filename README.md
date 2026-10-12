# An Intelligent System to Distinguish Benign from Malignant Skin Lesions

## Aim
To develop a deep learning system using transfer learning (ResNet50) that classifies dermoscopic skin lesion images as **benign** or **malignant**, and to deliver it as a tele-dermatology screening tool for early detection.

## Domain
Deep Learning & Computer Vision

## Keywords
skin, lesion, classification, transfer, learning, CNN

## Abstract
This work presents a transfer-learning CNN built on ResNet to distinguish benign from malignant skin lesions, addressing the problem of skin lesion classification through a data-driven approach. Data collected from dermoscopic image datasets is cleaned, transformed, and used to build and evaluate the underlying model. The outcome is delivered as a tele-dermatology screening tool, enabling practical adoption beyond the classroom.

## Methodology
1. **Data collection:** dermoscopic images (ISIC / Kaggle "Skin Cancer: Malignant vs Benign")
2. **Preprocessing:** resizing to 224x224, ResNet normalization, augmentation (flip, rotation, zoom, contrast)
3. **Model:** ResNet50 pretrained on ImageNet + custom classifier head (GlobalAveragePooling, Dropout, Dense)
4. **Training:** Phase 1 trains the new head; Phase 2 fine-tunes the last 30 ResNet layers at a low learning rate
5. **Evaluation:** accuracy, precision, recall, confusion matrix, ROC-AUC
6. **Deployment:** `predict_lesion()` function for single-image screening

## Tech Stack
Python, TensorFlow/Keras, ResNet50, scikit-learn, NumPy

## Dataset Structure
```
data/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## How to Run
```bash
pip install tensorflow scikit-learn numpy
python skin_lesion_resnet.py
```
A GPU (for example, Google Colab) is recommended for faster training.

## Results
| Metric | Value |
|--------|-------|
| Accuracy | __ % |
| Precision | __ |
| Recall (malignant) | __ |
| ROC-AUC | __ |

## Disclaimer
This tool is a screening aid only and is **not** a substitute for professional medical diagnosis. Please consult a dermatologist for any skin concern.

## Team
| Name | Roll No. | Guide |
|------|----------|-------|
| Gujjula Rebaka Rani | 23ME1A5464 | Mr. K. Kiran |
