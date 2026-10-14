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
2. **Preprocessing:** resizing to 224x224, ResNet50 `preprocess_input` normalization, augmentation (flip, rotation, zoom, contrast)
3. **Model:** ResNet50 pretrained on ImageNet + custom classifier head (GlobalAveragePooling, Dropout, Dense)
4. **Training:** Phase 1 trains the new head; Phase 2 fine-tunes the last 30 ResNet layers at a low learning rate
5. **Evaluation:** accuracy, precision, recall, confusion matrix, ROC-AUC
6. **Deployment:** single-image screening prediction

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
```
Open the notebook in Google Colab (Runtime > T4 GPU), download the dataset, and run all cells.

## Results
Evaluated on the held-out test set of **660 images** (360 benign, 300 malignant).

| Metric | Value |
|--------|-------|
| Accuracy | 87% |
| Precision (malignant) | 0.84 |
| Recall (malignant) | 0.87 |
| F1-score (malignant) | 0.85 |
| ROC-AUC | 0.95 |

**Confusion matrix**

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 311 | 49 |
| **Actual Malignant** | 40 | 260 |

The model correctly identifies 87% of malignant lesions, which suits a screening tool where missing a malignant case is costlier than a false alarm.

## Limitations
- Trained on dermoscopic images, so accuracy drops on ordinary camera photos
- Limited dataset size and no clinical validation
- Works as a screening aid and cannot replace biopsy or a dermatologist's diagnosis

## Future Scope
- Larger and more diverse datasets (ISIC full archive)
- Explainability with Grad-CAM heatmaps
- Web/mobile deployment (Streamlit/Flask) for tele-dermatology use

## Disclaimer
This tool is a screening aid only and is **not** a substitute for professional medical diagnosis. Please consult a dermatologist for any skin concern.

## Author
Name : Gujjula Rebaka Rani (23ME1A5464) Guide : Mr. K. Kiran 
