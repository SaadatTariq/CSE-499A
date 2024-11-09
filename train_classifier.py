# import pickle
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
#
#
# data_dict = pickle.load(open('./data.pickle', 'rb'))
#
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])
#
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#
# model = RandomForestClassifier()
#
# model.fit(x_train, y_train)
#
# y_predict = model.predict(x_test)
#
# score = accuracy_score(y_predict, y_test)
#
# print('{}% of samples were classified correctly !'.format(score * 100))
#
# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

import pickle
import numpy as np
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split

# Load data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Check for consistency in data shapes
expected_shape = len(data_dict['data'][0])
consistent_data = []
consistent_labels = []

for i, entry in enumerate(data_dict['data']):
    if len(entry) == expected_shape:
        consistent_data.append(entry)
        consistent_labels.append(data_dict['labels'][i])
    else:
        print(f"Inconsistent data shape at index {i}: {len(entry)}")

# Convert consistent data and labels to numpy arrays
data = np.array(consistent_data)
labels = np.array(consistent_labels)

# Ensure data has the correct shape (e.g., reshape according to feature count)
num_landmarks = expected_shape // 2  # Assuming each entry is (x, y) pairs
data = data.reshape(-1, num_landmarks, 2, 1)

# Print the data shape for verification
print("Data shape after reshaping:", data.shape)

# One-hot encode labels
labels = utils.to_categorical(labels, num_classes=28)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Adjust the CNN model to avoid negative dimensions
model = models.Sequential([
    layers.Input(shape=(num_landmarks, 2, 1)),  # Ensure this matches your reshaped data
    layers.Conv2D(32, (3, 2), activation='relu', padding='same'),  # Adjust kernel size
    layers.MaxPooling2D((2, 1), padding='same'),  # Adjust pooling size and padding

    layers.Conv2D(64, (3, 2), activation='relu', padding='same'),  # Adjust kernel size
    layers.MaxPooling2D((2, 1), padding='same'),  # Adjust pooling size and padding

    layers.Conv2D(128, (3, 2), activation='relu', padding='same'),  # Adjust kernel size
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(28, activation='softmax')  # 28 classes for the 28 labels
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

# Save the trained model as a .keras file
model.save('asl_cnn_model.keras')

print("Model trained and saved as 'asl_cnn_model.keras' successfully.")
