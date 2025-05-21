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
> If you wish to download the original dataset, you can find it on Kaggle:  
> [https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning)


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


---

## 📈 Results Summary

| Metric              | Value (Example) |
|---------------------|-----------------|
| Test Accuracy       | 93.2%           |
| Cohen Kappa Score   | 0.91            |
| Best Class Accuracy | Normal (95%)    |

> *NOTE: Actual performance may vary slightly due to training randomness.*

---

## 🧠 Future Improvements

- Hyperparameter tuning (learning rate, batch size)
- Experiment with EfficientNet, ResNet50
- Deploy using Flask or Streamlit for real-time predictions
- Apply Grad-CAM for visualizing what the model "sees"

---

## 💻 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

---

## 🧰 How to Run

```bash
# Clone the repository
git clone https://github.com/Itzraj786iul/ColonNet-Disease-Classifier.git
cd ColonNet-Disease-Classifier

# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook wce-cnn.ipynb

🙋‍♂️ Author
Raziullah Ansari
📍 NIT Raipur
[🔗 LinkedIn](https://www.linkedin.com/in/raziullah-ansari-8984431b6/)


---

## 🧾 Sample `requirements.txt` 

```txt
tensorflow
opencv-python
matplotlib
numpy
pandas
scikit-learn
seaborn
jupyter

