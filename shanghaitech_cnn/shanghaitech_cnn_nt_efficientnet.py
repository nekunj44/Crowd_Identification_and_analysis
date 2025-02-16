import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import EfficientNetB3
from sklearn.utils.class_weight import compute_class_weight
from scipy.io import loadmat

# Dataset Paths
datasets = [
    (r"C:\shanghaitech\ShanghaiTech\part_A\train_data\images", 
     r"C:\shanghaitech\ShanghaiTech\part_A\train_data\ground-truth"),
    (r"C:\shanghaitech\ShanghaiTech\part_B\train_data\images", 
     r"C:\shanghaitech\ShanghaiTech\part_B\train_data\ground-truth")
]

# Parameters
IMAGE_SIZE = (300, 300)
NUM_CLASSES = 3  # Multi-class classification (based on head count)
BATCH_SIZE = 16  # Reduced batch size for finer updates
EPOCHS = 10
LEARNING_RATE = 1e-5  # Lower learning rate

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

            # Load and preprocess the image
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0
            preprocessed_images.append(img_array)

            # Assign labels based on total dataset head count
            if head_count <= 110:
                labels.append(0)
            elif 110 < head_count <= 300:
                labels.append(1)
            else:
                labels.append(2)

# Convert to NumPy arrays
preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, labels, test_size=0.2, random_state=42, stratify=labels
)

# One-hot encode labels
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)

# Load EfficientNetB3 model without top layers (for transfer learning)
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# Freeze initial layers but allow fine-tuning on last few layers
for layer in base_model.layers[:-30]:  # Unfreeze last 30 layers
    layer.trainable = False

# Model definition
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Adjusted Dropout
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Compute class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training the model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

# Plotting Accuracy and Loss
def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(smooth_curve(history.history['accuracy']), label='Train Accuracy')
plt.plot(smooth_curve(history.history['val_accuracy']), label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(smooth_curve(history.history['loss']), label='Train Loss')
plt.plot(smooth_curve(history.history['val_loss']), label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision, Recall, F1-score
report = classification_report(true_classes, predicted_classes, target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)
