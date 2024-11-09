import os
import cv2

# Directory for saving collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3  # Number of classes for data collection
dataset_size = 100     # Number of images per class

# Initialize the video capture (try different indices if needed)
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if 0 doesn't work
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Collect data for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user to press 'q' to start data collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start capturing dataset_size number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the frame and save it
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

        # Exit loop if 'q' is pressed during data collection
        if cv2.waitKey(25) == ord('q'):
            print("Stopped by user.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
