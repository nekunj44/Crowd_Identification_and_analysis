import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPUs Available: {physical_devices}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

# Dataset Paths
train_datasets = [
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\train_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\train_data\ground-truth")
]

test_datasets = [
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_A\test_data\ground-truth"),
    (r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\images",
     r"C:\Users\ADMIN\Downloads\shanghaitech-20250214T111803Z-001\shanghaitech\ShanghaiTech\part_B\test_data\ground-truth")
]

# Parameters
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 2  # Binary classification (Low Density, High Density)

# Function to extract head count from ground truth files
def get_head_count(gt_file_path):
    gt_data = loadmat(gt_file_path)
    head_count = len(gt_data["image_info"][0][0][0][0][0])
    return head_count

# Preprocess images and labels
def preprocess_data(datasets):
    images, labels = [], []

    def numerical_sort(file_name):
        return int(file_name.split('_')[1].split('.')[0])

    for dataset_path, ground_truth_path in datasets:
        image_files = sorted(os.listdir(dataset_path), key=numerical_sort)

        for img_file in image_files:
            img_path = os.path.join(dataset_path, img_file)
            gt_file = os.path.join(ground_truth_path, f"GT_{os.path.splitext(img_file)[0]}.mat")

            if os.path.exists(gt_file):
                head_count = get_head_count(gt_file)

                # Load and preprocess the image
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(IMAGE_SIZE)
                img_array = np.array(img_resized) / 255.0
                images.append(img_array)

                # Assign labels based on head count
                if head_count <= 160:
                    labels.append(0)  # Low Density
                else:
                    labels.append(1)  # High Density

    return np.array(images), np.array(labels)

# Process training and test datasets
train_images, train_labels = preprocess_data(train_datasets)
test_images, test_labels = preprocess_data(test_datasets)

# One-hot encode labels
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)

# Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Training parameters
batch_size = 32
epochs = 20

# Training the model
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Plot training vs validation accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Density', 'High Density'], yticklabels=['Low Density', 'High Density'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
class_report = classification_report(true_labels, predicted_labels, target_names=['Low Density', 'High Density'])
print("Classification Report:\n", class_report)

# Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")