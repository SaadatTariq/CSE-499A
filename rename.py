import os

# Path to the main directory containing subfolders for each label
main_dir = '/Users/mdsaadattariq/PycharmProjects/CSE499A/ASL_Alphabet_Dataset/asl_alphabet_train'

# Iterate through each subfolder in the main directory
for folder_name in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Get all image file names sorted to ensure consistent order
        image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        # Rename the images sequentially from 0 to len(image_files) - 1
        for i, filename in enumerate(image_files):
            # Get the file extension
            _, file_extension = os.path.splitext(filename)

            # New file name with the format: '0.jpg', '1.jpg', ..., '1499.jpg'
            new_filename = f"{i}{file_extension}"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)

print("All images renamed successfully.")
