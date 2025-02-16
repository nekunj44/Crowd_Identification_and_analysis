import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # Added BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Added ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Dataset Path (Correct your paths if needed)
dataset_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\images"
labels_file_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\image_labels.txt"

# Parameters
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3

# Load Labels (Corrected)
labels_df = pd.read_csv(
    labels_file_path, 
    header=None, 
    names=["filename", "head_count", "scene_type", "weather", "distractor"],
    dtype={"filename": str}  # Important: Treat filename as string
)

labels_df = labels_df.iloc[:1500]

# Image Loading and Preprocessing (Corrected and More Efficient)
preprocessed_images = []
labels = []

for index, row in labels_df.iterrows(): # Iterate over the rows of the dataframe
    img_file = f"{row['filename']}.jpg"
    img_path = os.path.join(dataset_path, img_file)
    if os.path.exists(img_path):  # Check if image exists
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0
            preprocessed_images.append(img_array)

            head_count = int(row['head_count'])  # Convert to int directly

            if head_count <= 60:
                labels.append(0)
            elif 60 < head_count <= 200:
                labels.append(1)
            else:
                labels.append(2)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")  # Handle potential errors

preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, labels, test_size=0.2, random_state=42, stratify=labels # Added Stratification
)

# One-hot encode labels
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_labels = to_categorical(test_labels, NUM_CLASSES)

# CNN Model (Improved)
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), kernel_regularizer=l2(0.01)),  # Increased L2 regularization
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),  # Increased dropout

    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),  # Increased L2 regularization
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),  # Increased dropout

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Increased L2 regularization, reduced dense layer size
    BatchNormalization(),
    Dropout(0.5),  # Increased dropout
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks (Improved)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)  # Increased patience

# Train
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=10,  # Increased epochs
    batch_size=32,  # Increased batch size
    verbose=1,
    callbacks=[early_stopping, reduce_lr]  # Added ReduceLROnPlateau
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save
model.save("cnn_model.h5")

# Plotting training and validation loss/accuracy graphs
# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def predict_image(model_path, image_path):
    """Predict the class of a single image."""
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the input image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return predicted_class

# Path to the single image file
single_image_path = os.path.join(os.getcwd(), "test_image1.jpeg")  # Replace with your test image name

# Ensure the test image exists
if os.path.exists(single_image_path):
    predicted_label = predict_image("cnn_model.h5", single_image_path)
    print(f"Predicted Class for the test image: {predicted_label}")
else:
    print(f"Test image not found at {single_image_path}")