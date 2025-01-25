import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# Dataset Paths
dataset_path = r"C:\shanghaitech\ShanghaiTech\part_A\train_data\images"  # Adjust path for Part A or Part B
ground_truth_path = r"C:\shanghaitech\ShanghaiTech\part_A\train_data\ground-truth"  # Ground truth folder

# Parameters
IMAGE_SIZE = (224, 224)  # Image size for resizing
NUM_CLASSES = 3          # Multi-class classification (based on head count)

# Extract labels from ground truth
def get_head_count(gt_file_path):
    """
    Reads a ground truth file and returns the total head count.
    Assumes ground truth files are MATLAB .mat files with 'image_info'.
    """
    from scipy.io import loadmat
    gt_data = loadmat(gt_file_path)
    head_count = len(gt_data["image_info"][0][0][0][0][0])  # Number of points in the density map
    return head_count

# Preprocess images and labels
preprocessed_images = []  # List to store preprocessed images
labels = []  # List for storing labels based on head count

def numerical_sort(file_name):
    return int(file_name.split('_')[1].split('.')[0])

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
        preprocessed_images.append(img_array)

        # Assign labels based on head count
        if head_count <= 270:
            labels.append(0)
        elif head_count > 270 and head_count <= 490:
            labels.append(1)
        elif head_count > 490:
            labels.append(2)

# Convert to NumPy arrays
preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, labels, test_size=0.2, random_state=42
)

# One-hot encode labels
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)

# Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    
    Flatten(),  # Converts to a 1D vector to feed into Dense layers
    
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.6),
    Dense(3, activation='softmax')  # Output layer for classification
])

model.summary()


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
)

# Training parameters
batch_size = 29
epochs = 10

# Training the model without augmented data
history = model.fit(
    train_images, train_labels,
    batch_size=batch_size,
    validation_data=(test_images, test_labels),
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Plotting Accuracy and Loss
plt.figure(figsize=(12, 6))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
