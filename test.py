import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mediapipe as mp

# Load the trained CNN model
model = tf.keras.models.load_model('asl_cnn_model.keras')

# Path to the test images directory
TEST_DIR = '/Users/mdsaadattariq/PycharmProjects/CSE499A/ASL_Alphabet_Dataset/asl_alphabet_test'  # Replace with your actual path

# Label dictionary (adjust if needed)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Reverse mapping for label lookup
char_to_label = {v: k for k, v in labels_dict.items()}

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Process each image in the test directory
for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    data_aux = []
    x_ = []
    y_ = []

    # Extract the true label from the filename
    true_char = img_name.split('_')[0].upper()  # Assumes the format is "A_test.jpg"
    if true_char not in char_to_label:
        print(f"Skipping unrecognized label in filename: {img_name}")
        continue

    true_label = char_to_label[true_char]

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    # Process the image with Mediapipe
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Preprocess data for the CNN model
        data_aux = np.array(data_aux).reshape(1, 21, 2, 1)  # Adjust based on the expected input shape
        prediction = model.predict(data_aux)
        predicted_label = np.argmax(prediction)

        # Append true and predicted labels
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

# Check if the labels were populated correctly
if not true_labels or not predicted_labels:
    print("No data was processed for evaluation. Check your test image paths and processing steps.")
    exit(1)

# Generate and print evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=list(labels_dict.keys()), target_names=list(labels_dict.values())))

hands.close()
