import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping

# Dataset Path
dataset_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\images"
labels_file_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\train\image_labels.txt"

# Parameters
IMAGE_SIZE = (224, 224)  # Image size for resizing
NUM_CLASSES = 3       # Binary classification (update based on your task)

# Read the labels file, ensuring 'filename' is treated as a string
labels_df = pd.read_csv(
    labels_file_path, 
    header=None, 
    names=["filename", "head_count", "scene_type", "weather", "distractor"],
    dtype={"filename": str}  # Explicitly treat the filename column as string
)

# Sort the filenames in the dataset directory
sorted_files = sorted(os.listdir(dataset_path))

# Filter and sort labels from the DataFrame
sorted_labels = sorted(labels_df["filename"])

# Preprocessing images
image_files = [f"{img_file}.jpg" for img_file in labels_df["filename"] if f"{img_file}.jpg" in os.listdir(dataset_path)][:1000]

preprocessed_images = []  # List to store preprocessed images
labels = []  # List for storing labels based on head count
        
# Process images and assign labels
for img_file in image_files:
    img_path = os.path.join(dataset_path, img_file)
    img_file_without_extension = img_file.split('.')[0]

    if img_file_without_extension in labels_df["filename"].values:
        head_count = labels_df.loc[labels_df["filename"] == img_file_without_extension, "head_count"].values[0]
        head_count = int(head_count)  # Ensure head_count is an integer

        img = Image.open(img_path).convert('RGB')  # Convert to RGB
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
        preprocessed_images.append(img_array)

        # Assign binary label based on head count
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

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    Dropout(0.1),

    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3), 

    Flatten(),  # Flatten the feature maps
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.7),  # Dropout to reduce overfitting
    Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # You can also use 'val_accuracy' instead
                               patience=3,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restores the best weights once training stops
                               verbose=1)

# Train the model with early stopping
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=10,  # Number of epochs
    batch_size=16,  # Batch size
    verbose=1,
    callbacks=[early_stopping]  # Add the EarlyStopping callback here
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model
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

# -------------------------------
# Test the model on a single image
# -------------------------------

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
