###################################################################################################
# THIS IS THE CODE WHICH USED TO CONVERT SIGN LANGUAGE ACTION TO NUMPY DATA
###################################################################################################
import mediapipe as mp
import numpy as np

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