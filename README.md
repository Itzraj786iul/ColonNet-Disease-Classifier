# GI-Disease-Classifier 🧠🔬

A deep learning-based image classification project to detect and categorize gastrointestinal diseases from endoscopic images using Convolutional Neural Networks (CNNs) and transfer learning with MobileNet.

---

## 📌 Project Overview

This project focuses on classifying gastrointestinal diseases into the following four categories:
- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

The goal is to assist medical professionals in faster and more consistent diagnoses using automated image classification models.

---

## 🚀 Model Highlights

- **Base Model:** MobileNet (with ImageNet weights, `include_top=False`)
- **Architecture:** MobileNet backbone + Conv2D + MaxPooling2D + BatchNormalization + Dense Layers
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Evaluation:** Accuracy, ROC Curves, Cohen Kappa Score, Classification Report

---

## 📂 Dataset

- **Source:** Custom dataset (4-class balanced endoscopic image dataset)
- **Structure:**

| Split       | Normal | Ulcerative Colitis | Polyps | Esophagitis |
|-------------|--------|--------------------|--------|-------------|
| Training    | 800    | 800                | 800    | 800         |
| Validation  | 500    | 500                | 500    | 500         |
| Test        | 200    | 200                | 200    | 200         |

## Organized in folders:
-data/
-├── train/
-├── val/
-└── test/

---

## 🧪 Data Augmentation

Applied real-time image augmentation using Keras ImageDataGenerator:
- Rotation Range: ±30°
- Zoom Range: 20%
- Width/Height Shift: 20%
- Horizontal & Vertical Flip
- Rescaling (1./255)

---

## 🧠 Model Architecture

```python
def create_bir_model():
    base_model = MobileNet(include_top=False, weights='imagenet')
    model = Sequential([
        base_model,
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model 
```

## 📊 Evaluation Metrics

### ✅ Deep Learning (MobileNet-based Model)

| Metric           | Value   |
|------------------|---------|
| Test Accuracy    | 98%     |
| Validation Loss  | 0.1483  |
| Cohen's Kappa    | 0.977   |

### 📈 Classical Machine Learning (10-Fold CV)

| Model               | Accuracy |
|---------------------|----------|
| SVM                 | 99.44%   |
| Random Forest       | 99.38%   |
| Logistic Regression | 99.38%   |
| Decision Tree       | 98.56%   |

---

## 📁 Project Structure

├── data/
│ ├── train/
│ ├── val/
│ └── test/
├── models/
├── requirements.txt
└── README.md



---

## 💻 Technologies Used

- Python 3.8+
- TensorFlow 2.12
- OpenCV 4.7
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/gi-disease-classification.git
cd gi-disease-classification
```
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py \
  --train_dir path/to/train \
  --val_dir path/to/val \
  --epochs 30 \
  --batch_size 32


## 📈 Future Improvements

- Use EfficientNet, DenseNet, or ResNet variants  
- Experiment with learning rate schedulers and optimizers  
- Deploy via Streamlit/Flask for real-time inference  
- Integrate Grad-CAM for interpretability  

---

## 🙋‍♂️ Author

**Raziullah Ansari**  
📍 NIT Raipur  
🔗 [LinkedIn](https://www.linkedin.com/in/raziullah-ansari-8984431b6/)
