# An Intelligent System to Distinguish Benign from Malignant Skin Lesions

## Aim
To develop a deep learning system using transfer learning (ResNet50) that classifies dermoscopic skin lesion images as **benign** or **malignant**, as a prototype tele-dermatology screening tool for early detection.

## Domain
Deep Learning & Computer Vision

## Keywords
skin, lesion, classification, transfer, learning, CNN

## Abstract
This work presents a transfer-learning CNN built on ResNet50 to distinguish benign from malignant skin lesions, addressing the problem of skin lesion classification through a data-driven approach. Data collected from dermoscopic image datasets is cleaned, transformed, and used to build and evaluate the underlying model. The outcome is a screening model that can be extended into a tele-dermatology tool, enabling practical adoption beyond the classroom.

## Methodology
1. **Data collection:** dermoscopic images (ISIC-based Kaggle dataset "Skin Cancer: Malignant vs Benign")
2. **Preprocessing:** resizing to 224x224, ResNet50 `preprocess_input` normalization, augmentation (flip, rotation, zoom, contrast)
3. **Model:** ResNet50 pretrained on ImageNet + custom classifier head (GlobalAveragePooling, Dropout 0.3, Dense sigmoid)
4. **Training:** Phase 1 trains the new head; Phase 2 fine-tunes the last 30 ResNet layers at a low learning rate (Adam, 1e-5)
5. **Evaluation:** accuracy, precision, recall, confusion matrix, ROC-AUC
6. **Prediction:** single-image screening (benign / malignant with probability)

## Tech Stack
Python, TensorFlow/Keras, ResNet50, scikit-learn, NumPy, Matplotlib

## Dataset
[Skin Cancer: Malignant vs Benign (Kaggle)](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) - 2,637 training and 660 test images (224x224). 20% of the training images are used for validation (seed 42). The dataset is not stored in this repository.
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
pip install -r requirements.txt
kaggle datasets download -d fanconic/skin-cancer-malignant-vs-benign
unzip -q skin-cancer-malignant-vs-benign.zip -d data
python skin_cancer_classification.py
```
Or open the notebook in Google Colab (Runtime > T4 GPU) and run all cells.

## Results
Evaluated on the held-out test set of **660 images** (360 benign, 300 malignant).

| Metric | Value |
|--------|-------|
| Accuracy | 87.0% (574 / 660) |
| Precision (malignant) | 0.86 |
| Recall (malignant) | 0.85 |
| F1-score (malignant) | 0.86 |
| Specificity (benign recall) | 0.89 |
| ROC-AUC | 0.95 |

**Confusion matrix**

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 320 | 40 |
| **Actual Malignant** | 46 | 254 |

The model correctly identifies 85% of malignant lesions (254 of 300) and misses 46, which is the main weakness for a screening tool where a missed malignant case is costlier than a false alarm. Lowering the decision threshold below 0.5 can raise recall at the cost of more false alarms. Training accuracy (~95%) is higher than validation accuracy (~87%), indicating some overfitting.

![Training curves, confusion matrix and ROC curve](results/results.png)

## Limitations
- Trained on dermoscopic images, so accuracy drops on ordinary camera photos
- Small dataset (~3,300 images) and no clinical validation
- Some overfitting and a 15% miss rate for malignant lesions
- Works as a screening aid and cannot replace biopsy or a dermatologist's diagnosis

## Future Scope
- Larger and more diverse datasets (ISIC full archive)
- Explainability with Grad-CAM heatmaps
- Web/mobile deployment (Streamlit/Flask) for tele-dermatology use

## Trained Model
`skin_cancer_model.keras` is about 210 MB (above GitHub's 100 MB limit), so it is published under **Releases**: <paste release link here>

## Disclaimer
This tool is a screening aid only and is **not** a substitute for professional medical diagnosis. Please consult a dermatologist for any skin concern.

## Author
**Name:** Gujjula Rebaka Rani (23ME1A5464)
**Guide:** Dr. P. Sudhakar
