import os
import random
import cv2
import shutil

# Path to the main directory containing subfolders for each alphabet
main_dir = '/Users/mdsaadattariq/PycharmProjects/CSE499A/ASL_Alphabet_Dataset/asl_alphabet_train'

# Define the label encoding for subfolder names
labels = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'DEL': 26, 'SPACE': 27
}


# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return False
        return True
    except Exception as e:
        print(f"Skipped corrupted image: {image_path}, Error: {e}")
        return False


# Iterate through subfolders in the main directory
for folder_name in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder_name)

    # Check if it's a directory and a valid label
    if os.path.isdir(folder_path) and folder_name in labels:
        # Get all image file names and filter valid images
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        valid_images = [f for f in image_files if is_image_valid(os.path.join(folder_path, f))]

        # Reduce the number of images to 1500 if needed
        if len(valid_images) > 1500:
            valid_images = random.sample(valid_images, 1500)

        # Keep only the valid and selected images
        for file in image_files:
            if file not in valid_images:
                os.remove(os.path.join(folder_path, file))

        # Rename the folder to its encoded label
        new_folder_name = str(labels[folder_name])
        new_folder_path = os.path.join(main_dir, new_folder_name)
        os.rename(folder_path, new_folder_path)

print("Folders processed, valid images kept, and folders renamed successfully.")
