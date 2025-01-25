# 🏙️ Crowd Density Classification using CNNs 🤖

Welcome to the **Crowd Density Classification** project! 🚀 This repository provides deep learning solutions using **CNN-based models** to classify crowd density levels in images from the **JHU-CROWD++** and **ShanghaiTech** datasets. 📷

---

## 📂 Project Overview

We present solutions for crowd density estimation using two widely known datasets:

- **JHU-CROWD++ Dataset 🏛️**
  - Classifies images into three density levels based on head counts.
  - Utilizes convolutional neural networks with **dropout** and **early stopping** to enhance performance.

- **ShanghaiTech Dataset 🏢**
  - Provides detailed head count data via ground truth annotations (.mat files).
  - Includes **preprocessing**, **CNN training**, and **visualization**.

---

## 🚀 Features

✨ **End-to-End Pipeline:** Image preprocessing, training, evaluation, and prediction.

✨ **CNN Architecture:** Multi-layered convolutional networks optimized for crowd analysis.

✨ **Early Stopping:** Prevents overfitting by monitoring validation performance.

✨ **Learning Rate Reduction:** Automatically adjusts learning rates to improve convergence.

✨ **Visualization:** Accuracy and loss graphs to track progress.

✨ **Customizable:** Modify dataset paths and parameters with ease.

---

## 🛠️ Prerequisites

Ensure the following dependencies are installed before running the project:

**Python 3.x 🐍**

### Required Libraries:

```bash
pip install tensorflow numpy pandas matplotlib pillow scipy
```

---

## 📁 Project Structure

```
crowd-density-classification/
│-- JHU_CROWD++/           # Files related to JHU-CROWD++ dataset
│   ├── jhucrowdpp_cnn.py   # Model training and evaluation script
│   ├── Readme1.md          # JHU dataset files (images and labels)
│-- ShanghaiTech/          # Files related to ShanghaiTech dataset
│   ├── shanghaitech_cnn.py # Model training and evaluation script
│   ├── Readme2.md          # ShanghaiTech dataset files (images and labels)
│-- README.md              # Project documentation
```

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/crowd-density-classification.git
cd crowd-density-classification
```

### 2️⃣ Modify Dataset Paths

Update dataset paths in respective scripts:

```python
# JHU-CROWD++ dataset
dataset_path = r"C:\JHU_CROWD++_DATASET\train\images"
labels_file_path = r"C:\JHU_CROWD++_DATASET\train\image_labels.txt"

# ShanghaiTech dataset
dataset_path = r"C:\shanghaitech\part_A\train_data\images"
ground_truth_path = r"C:\shanghaitech\part_A\train_data\ground-truth"
```

### 3️⃣ Train the Models 🏋️‍♂️

#### For JHU-CROWD++ Dataset:

```bash
python JHU_CROWD++/jhucrowdpp_cnn.py
```

#### For ShanghaiTech Dataset:

```bash
python ShanghaiTech/shanghaitech_cnn.py
```

### 4️⃣ Evaluate the Model 🧪

Upon training completion, accuracy and loss metrics will be displayed.

### 5️⃣ Make Predictions 🔮

```python
from cnn_model import predict_image

predicted_label = predict_image("cnn_model.h5", "test_image.jpeg")
print(f"Predicted Crowd Density Class: {predicted_label}")
```

---

## 🏗️ Model Architectures

### JHU-CROWD++ CNN Model 🏛️

- **Input:** 224x224 RGB Images
- **Layers:**
  - Conv2D (32 filters, 3x3 kernel) ➡️ MaxPooling (2x2) ➡️ Dropout (0.1)
  - Conv2D (64 filters, 3x3 kernel) ➡️ MaxPooling (2x2) ➡️ Dropout (0.2)
  - Conv2D (128 filters, 3x3 kernel) ➡️ MaxPooling (2x2) ➡️ Dropout (0.3)
  - Fully Connected Layer ➡️ Dropout (0.7) ➡️ Output (Softmax for 3 classes)

### ShanghaiTech CNN Model 🏢

- **Input:** 224x224 RGB Images
- **Layers:**
  - Conv2D (32 filters, 3x3 kernel) ➡️ MaxPooling (2x2) ➡️ Dropout (0.3)
  - Conv2D (64 filters, 3x3 kernel) ➡️ MaxPooling (2x2) ➡️ Dropout (0.4)
  - Fully Connected Layer ➡️ Dropout (0.6) ➡️ Output (Softmax for 3 classes)

---

## 📊 Results

- Both models achieve **satisfactory performance** on their respective test sets.
- Training stops early if no improvements are observed.

### 📈 Visualization

Training and validation accuracy/loss graphs are plotted for better analysis.

---

## ⚡ Performance Improvements

The following techniques are employed to enhance performance:

- **Dropout:** Prevents overfitting during training.
- **Early Stopping:** Stops training if validation loss stops improving.
- **Learning Rate Reduction:** Adjusts learning rates based on validation performance.
- **Batch Normalization:** Helps with faster convergence and improved accuracy.

---

## 📝 Acknowledgments

Special thanks to the creators of the **JHU-CROWD++** and **ShanghaiTech** datasets. 🙏

---

## 📧 Contact

For any questions, feel free to reach out:

- 📩 Email: [nekunj44@gmail.com](mailto:nekunj44@gmail.com)
- 🌐 GitHub: [nekunj44](https://github.com/nekunj44)

---

🌟 **Don't forget to star this repo if you found it useful!** ⭐