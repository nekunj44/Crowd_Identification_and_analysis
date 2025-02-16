import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Dataset Path
dataset_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\images"
labels_file_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\image_labels.txt"

# Parameters
IMAGE_SIZE = (224, 224)  # Image size for EfficientNet
NUM_CLASSES = 3  # Number of classes
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

# Read the labels file
labels_df = pd.read_csv(
    labels_file_path, 
    header=None, 
    names=["filename", "head_count", "scene_type", "weather", "distractor"],
    dtype={"filename": str}
)

# Filter and sort image files
image_files = [f"{img_file}.jpg" for img_file in labels_df["filename"] if f"{img_file}.jpg" in os.listdir(dataset_path)][:2000]

# Preprocessing images and labels
preprocessed_images = []
labels = []

for img_file in image_files:
    img_path = os.path.join(dataset_path, img_file)
    img_file_without_extension = img_file.split('.')[0]

    if img_file_without_extension in labels_df["filename"].values:
        head_count = labels_df.loc[labels_df["filename"] == img_file_without_extension, "head_count"].values[0]
        head_count = int(head_count)  # Ensure head_count is an integer

        # Preprocess the image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        preprocessed_images.append(img_array)

        # Assign labels based on head count
        if head_count <= 60:
            labels.append(0)
        elif head_count > 60 and head_count <= 200:
            labels.append(1)
        elif head_count > 200:
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

# Load EfficientNetB0 model as base
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the first 210 layers
for layer in base_model.layers[:210]:
    layer.trainable = False

# Add classification head
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.01))(x)

# Build the model
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("efficientnet_model.h5")

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predictions for evaluation
y_pred_prob = model.predict(test_images)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

# Convert one-hot encoded test_labels back to class labels
test_labels_classes = np.argmax(test_labels, axis=1)

# Generate confusion matrix and classification metrics
conf_matrix = confusion_matrix(test_labels_classes, y_pred)
class_report = classification_report(test_labels_classes, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1-Score
class_report = classification_report(test_labels, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
print("Classification Report:\n", class_report)

# Single Image Prediction
def predict_image(model_path, image_path):
    """Predict the class of a single image."""
    model = load_model(model_path)

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Test single image prediction
single_image_path = os.path.join(os.getcwd(), "test_image1.jpeg")  # Update with your test image path
if os.path.exists(single_image_path):
    predicted_label = predict_image("efficientnet_model.h5", single_image_path)
    print(f"Predicted Class for the test image: {predicted_label}")
else:
    print(f"Test image not found at {single_image_path}")