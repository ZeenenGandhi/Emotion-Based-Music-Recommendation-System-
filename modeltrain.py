import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialize variables
initialized = False
data_size = -1
labels = []
label_dict = {}
label_counter = 0

# Load data
for file in os.listdir():
    if file.endswith(".npy") and not file.startswith("labels"):  
        if not initialized:
            initialized = True 
            X_data = np.load(file)
            data_size = X_data.shape[0]
            y_data = np.array([file.split('.')[0]] * data_size).reshape(-1, 1)
        else:
            X_data = np.concatenate((X_data, np.load(file)))
            y_data = np.concatenate((y_data, np.array([file.split('.')[0]] * data_size).reshape(-1, 1)))

        labels.append(file.split('.')[0])
        label_dict[file.split('.')[0]] = label_counter  
        label_counter += 1

# Encode labels
for i in range(y_data.shape[0]):
    y_data[i, 0] = label_dict[y_data[i, 0]]
y_data = np.array(y_data, dtype="int32")
y_data = to_categorical(y_data)

# Shuffle data
X_shuffled = X_data.copy()
y_shuffled = y_data.copy()
shuffle_indices = np.arange(X_data.shape[0])
np.random.shuffle(shuffle_indices)
for idx, shuffle_idx in enumerate(shuffle_indices): 
    X_shuffled[idx] = X_data[shuffle_idx]
    y_shuffled[idx] = y_data[shuffle_idx]

# Define model
input_layer = Input(shape=(X_data.shape[1]))
dense_layer_1 = Dense(512, activation="relu")(input_layer)
dense_layer_2 = Dense(256, activation="relu")(dense_layer_1)
output_layer = Dense(y_data.shape[1], activation="softmax")(dense_layer_2) 
emotion_model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
emotion_model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

# Train model
emotion_model.fit(X_shuffled, y_shuffled, epochs=50)

# Save model and labels
emotion_model.save("model.h5")
np.save("labels.npy", np.array(labels))