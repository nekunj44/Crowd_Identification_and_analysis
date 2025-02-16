import scipy.io

# Define the file path
file_path = r"C:\shanghaitech\ShanghaiTech\part_B\train_data\ground-truth\GT_IMG_1.mat"

# Load the .mat file
data = scipy.io.loadmat(file_path)

# Display the keys in the file
print("Keys in the .mat file:", data.keys())

# Assuming the ground truth data is stored under a key like 'image_info' or similar
for key in data:
    if not key.startswith("__"):  # Ignore meta keys
        print(f"\nContents of key '{key}':")
        print(data[key])
