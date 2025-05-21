# ColonNet-Disease-Classifier 🧠🔬

A deep learning-based image classification project to detect and categorize colon-related diseases from endoscopic images using Convolutional Neural Networks (CNNs) and transfer learning with MobileNet.

---

## 📌 Project Overview

The project focuses on building a robust CNN-based model to classify images of colonoscopy into four categories:
- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

Medical professionals often rely on endoscopic imaging for diagnosis. Automating classification can assist in faster and more consistent diagnoses, particularly in resource-limited settings.

---

## 🚀 Model Highlights

- **Base Model:** MobileNet (with ImageNet weights, `include_top=False`)
- **Architecture:** MobileNet backbone + Conv2D + MaxPooling2D + BatchNormalization + Dense Layers
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Evaluation:** Accuracy, Confusion Matrix, Classification Report, Cohen Kappa Score

---

## 📂 Dataset

**Source:** [Kaggle - Curated Colon Dataset for Deep Learning](https://www.kaggle.com/datasets/satwikakth/curated-colon-dataset-for-deep-learning)

- 4 Classes
- Pre-organized into:
  - `train/`
  - `test/`
  - `val/`
- Real-world colonoscopy images

---

## 🧪 Data Augmentation

Applied real-time augmentation to improve generalization:
- Rotation Range: ±30°
- Zoom Range: 20%
- Width/Height Shift: 20%
- Horizontal & Vertical Flip
- Rescaling (1./255)

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Cohen Kappa Score**
- **Precision, Recall, F1-score (via `classification_report`)**

---

## 📁 Folder Structure
ColonNet-Disease-Classifier/
│
├── wce-cnn.ipynb # Jupyter Notebook with model code
├── README.md # Project documentation
├── requirements.txt # Python dependencies (optional)
└── dataset/ # Link or info to Kaggle dataset
