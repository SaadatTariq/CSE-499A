import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the dataset directory
DATA_DIR = '/Users/mdsaadattariq/Downloads/sign-language-detector-python-master/data'

# Lists to store data and labels
data = []
labels = []

# Iterate through 27 subfolders (labeled as 0 to 27)
for i in range(28):  # 0 to 27 inclusive
    dir_path = os.path.join(DATA_DIR, str(i))
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None:
                continue  # Skip corrupted or unreadable images

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image with Mediapipe
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect x and y coordinates for normalization
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    # Normalize and store the data
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(i)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()
print("Data collection and saving completed.")
