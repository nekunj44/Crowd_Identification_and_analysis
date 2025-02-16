import os
import numpy as np
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# Dataset Paths
datasets = [
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\ground-truth")
]

# Function to extract head count from ground truth files
def get_head_count(gt_file_path):
    gt_data = loadmat(gt_file_path)
    head_count = len(gt_data["image_info"][0][0][0][0][0])
    return head_count

# Preprocess images and labels
preprocessed_images = []
labels = []

def numerical_sort(file_name):
    return int(file_name.split('_')[1].split('.')[0])

for dataset_path, ground_truth_path in datasets:
    image_files = sorted(os.listdir(dataset_path), key=numerical_sort)

    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        gt_file = os.path.join(ground_truth_path, f"GT_{os.path.splitext(img_file)[0]}.mat")

        if os.path.exists(gt_file):
            head_count = get_head_count(gt_file)

            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0
            preprocessed_images.append(img_array)

            if head_count <= 110:
                labels.append(0)
            elif 110 < head_count <= 300:
                labels.append(1)
            else:
                labels.append(2)

preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, labels, test_size=0.2, random_state=42, stratify=labels#initially 0.2 thaa
)

train_labels_categorical = tf.keras.utils.to_categorical(train_labels, 3)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels, 3)

# Model Definition
def mobilenet_v2_model(input_shape):
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs, x)
    return model

model = mobilenet_v2_model((128, 128, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),#initisally 1e-4
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Enable GPU Mirrored Strategy for multi-GPU training if available
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = mobilenet_v2_model((128, 128, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

history = model.fit(
    train_images, train_labels_categorical, batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(test_images, test_labels_categorical),
    callbacks=[early_stopping]
)

# Evaluation
eval_result = model.evaluate(test_images, test_labels_categorical, batch_size=BATCH_SIZE)
print(f"Test Accuracy: {eval_result[1]:.2f}")

# Predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_labels

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=["Low", "Medium", "High"])
print("Classification Report:\n", report)

# Plot Accuracy and Loss
def plot_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history)