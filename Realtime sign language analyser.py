###################################################################################################
# THIS IS THE CODE OF REAL-TIME ANALYZER
###################################################################################################
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import cv2
from PIL import Image
import mediapipe as mp

# Load the trained LSTM model
model = load_model('trained_model_LSTM.h5')

# Function to recognize sign language gesture from an image
def recognize_sign_language(ND_list):
    
    predictions = model.predict(np.array([ND_list]))
    predicted_label = np.argmax(predictions)

    return predicted_label

def encoder(image):
    mp_holistic = mp.solutions.holistic

    # Model setting
    with mp_holistic.Holistic(model_complexity=2,
                              smooth_landmarks=True,
                              enable_segmentation=True,
                              min_detection_confidence=0.8, 
                              min_tracking_confidence=0.8) as model:
        
        results = model.process(np.array(image))

        # Landmark data
        lh_data = np.array([[res.x, res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh_data = np.array([[res.x, res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        pose_data = np.array([[res.x, res.y,res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # Concatenate the arrays and reshape to a consistent shape
        data = np.concatenate([pose_data, lh_data, rh_data])
        return data

if __name__ == "__main__":
    # Open the webcam
    cap = cv2.VideoCapture(0)
    # Set the video frame rate to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    ND_list=[]
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        predicted_label=0
        image_data = encoder(frame)
        if len(ND_list)==30:
            ND_list.remove(0)
            ND_list.append(image_data)
            
            # Recognize the sign language gesture in the image
            predicted_label = recognize_sign_language(ND_list)
            
        else:ND_list.append(image_data)

        # Define the mapping of labels to sign language gestures
        # label_mapping = {0: 'goodafternoon',1: 'goodmorning',2: 'goodnight',3: 'thankyou'}
        label_mapping = {0: 'goodafternoon',1: 'goodmorning',2: 'goodnight',3: 'thankyou'}
        # Get the corresponding sign language gesture based on the predicted label
        predicted_gesture = label_mapping[predicted_label]
        
        # Display the predicted sign language gesture on the frame
        cv2.putText(frame, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Add a delay to achieve 30 fps
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()