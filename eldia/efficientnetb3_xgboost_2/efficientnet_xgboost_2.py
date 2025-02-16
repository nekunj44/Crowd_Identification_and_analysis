import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import xgboost as xgb
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Dataset Paths
datasets_train = [
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\ground-truth")
]

datasets_test = [
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\ground-truth")
]

# Parameters
IMAGE_SIZE = (300, 300)
NUM_CLASSES = 2  # Low, High crowd density classes

# Extract head count from ground truth
def get_head_count(gt_file_path):
    gt_data = loadmat(gt_file_path)
    head_count = len(gt_data["image_info"][0][0][0][0][0])
    return head_count

# Load EfficientNetB3 as feature extractor
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
global_avg_layer = GlobalAveragePooling2D()

# Preprocess images and extract features
def load_features_and_labels(dataset_paths):
    features = []
    labels = []
    for dataset_path, ground_truth_path in dataset_paths:
        image_files = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        for img_file in image_files:
            img_path = os.path.join(dataset_path, img_file)
            gt_file = os.path.join(ground_truth_path, f"GT_{os.path.splitext(img_file)[0]}.mat")

            if os.path.exists(gt_file):
                head_count = get_head_count(gt_file)

                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(IMAGE_SIZE)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                feature = base_model.predict(img_array, verbose=0)
                pooled_feature = global_avg_layer(feature).numpy().flatten()

                features.append(pooled_feature)

                if head_count <= 160:
                    labels.append(0)  # Low Density
                else:
                    labels.append(1)  # High Density
    return np.array(features), np.array(labels)

# Load training and testing data
train_features, train_labels = load_features_and_labels(datasets_train)
test_features, test_labels = load_features_and_labels(datasets_test)

# Train XGBoost classifier with GPU support and early stopping
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric=['logloss', 'error'],
    learning_rate=0.01,
    n_estimators=50,
    max_depth=4,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=1.0,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    device='cuda'
)

history = xgb_model.fit(
    train_features, train_labels,
    eval_set=[(train_features, train_labels), (test_features, test_labels)],
    verbose=True
)

# Plot Accuracy and Loss
results = xgb_model.evals_result()
epochs = range(1, len(results['validation_0']['logloss']) + 1)

plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, results['validation_0']['logloss'], label='Train Loss')
plt.plot(epochs, results['validation_1']['logloss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
train_accuracy = [1 - x for x in results['validation_0']['error']]
val_accuracy = [1 - x for x in results['validation_1']['error']]
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
accuracy = xgb_model.score(test_features, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred = xgb_model.predict(test_features)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred)

# Compute classification metrics
report = classification_report(test_labels, y_pred, target_names=['Low Density', 'High Density'])

# Print metrics
print("Classification Report:\n", report)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
