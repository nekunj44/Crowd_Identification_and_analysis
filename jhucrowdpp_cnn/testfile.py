import os
import pandas as pd

# Dataset Paths
dataset_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\test\images"
labels_file_path = r"C:\JHU_CROWD++_DATASET\jhu_crowd_v2.0\test\image_labels.txt"

# Read labels file
labels_df = pd.read_csv(
    labels_file_path, 
    header=None, 
    names=["filename", "head_count", "scene_type", "weather", "distractor"],
    dtype={"filename": str}
)

# Get first 1500 valid image filenames
image_files = [
    f"{img_file}.jpg" for img_file in labels_df["filename"] if f"{img_file}.jpg" in os.listdir(dataset_path)
][:4000]

# Initialize class counters
class_counts = {0: 0, 1: 0, 2: 0}

# Count images per class
for img_file in image_files:
    img_file_without_extension = img_file.split('.')[0]

    if img_file_without_extension in labels_df["filename"].values:
        head_count = labels_df.loc[labels_df["filename"] == img_file_without_extension, "head_count"].values[0]
        head_count = int(head_count)  # Ensure head_count is an integer

        # Assign class and update count
        if head_count <= 60:
            class_counts[0] += 1
        elif 60 < head_count <= 200:
            class_counts[1] += 1
        elif head_count > 200:
            class_counts[2] += 1

# Print results
print("Image count per class:")
print(f"Class 0 (≤60 people): {class_counts[0]}")
print(f"Class 1 (61-200 people): {class_counts[1]}")
print(f"Class 2 (>200 people): {class_counts[2]}")
