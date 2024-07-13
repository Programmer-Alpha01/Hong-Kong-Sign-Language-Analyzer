###################################################################################################
# THIS IS THE CODE FOR THE LSTM NETWORK
###################################################################################################

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Bidirectional,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_PATH = os.path.join('training_data') 
actions = np.array(['goodafternoon', 'goodmorning', 'goodnight', 'thankyou'])
label_map = {label:num for num, label in enumerate(actions)}

Number_of_folders_per_actions=470
Number_of_npy_files_per_folders=30
shape=258
sequences, labels = [], []
x=len(actions)
process=0

# Load the data
for action in actions:
    for sequence in range(Number_of_folders_per_actions):
        window = []
        for frame_num in range(Number_of_npy_files_per_folders):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
    process+=1
    print(f"{(process/x)*100}%")

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)

 # Utilize GPU for training
with tf.device('/GPU:0'): 
    log_dir = os.path.join('Logs')
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(Number_of_npy_files_per_folders,shape)))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=False, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=100,shuffle=True,verbose=1,validation_data=[X_test,y_test],validation_split=0.2)

# Model save
model.save('trained_model_LSTM.h5')
print("Trained model saved as 'trained_model_LSTM.h5'")