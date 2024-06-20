###################################################################################################
# THIS IS THE CODE OF SIGN LANGUAGE VIDEO ANALYZER
###################################################################################################
import mediapipe as mp
import numpy as np
import cv2
import os
import glob
from moviepy.editor import VideoFileClip, concatenate, vfx
import keras
from keras.models import load_model
from PIL import Image
import time

model=keras.models.load_model('trained_model_LSTM.h5')
mp_holistic = mp.solutions.holistic

def change_frame_rate(input_video, output_video, target_frame_rate):
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, target_frame_rate, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        out.write(frame)
    print(f"{cap} Convert complete !")
    cap.release()
    out.release()

def adjust_video_length(video, Buffer, target_duration=1):
    clip = VideoFileClip(video)
    duration = clip.duration
    if duration < target_duration:
        clip = VideoFileClip(video)
        extension_clip = clip.subclip(clip.duration - target_duration)
        extended_clip = concatenate([clip, extension_clip])
        extended_clip.write_videofile(Buffer)
        clip.close()
        extension_clip.close()
        extended_clip.close()

    elif duration > target_duration:
        clip = VideoFileClip(video)
        clip_duration = clip.duration
        compression_factor = clip_duration / target_duration
        compressed_clip = clip.fx(vfx.speedx, compression_factor)
        compressed_clip.write_videofile(Buffer)
        clip.close()
        compressed_clip.close()

def encoder(image):

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

def landmark(video):
    list=[]
    video = cv2.VideoCapture(video)
    while True:
        ret,frame=video.read()
        if not ret:break
        else:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_data = encoder(frame_pil)
            list.append(frame_data)
    return list

def prediction(data):
    
    predictions = model.predict(np.array([data]))
    predicted_label = np.argmax(predictions)
    return predicted_label

if __name__ == "__main__":
    startTime = time.time()
    folder_path = 'input_video'
    Buffer='Buffer'
    Buffer1='Buffer/Buffer1'
    Buffer2='Buffer/Buffer2'

    # Setting
    target_frame_rate=30
    target_duration=1

    # Build buffers
    os.makedirs(Buffer1, exist_ok=True)
    os.makedirs(Buffer2, exist_ok=True)

    print("Start to analyze Sign Language Video")
    

    print("\nConverting video frame rate")
    print("Process complete: 0%\n")

    # Process each video file
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    for video in video_files:
        Buffer = os.path.join(Buffer1, os.path.basename(video))
        change_frame_rate(video, Buffer, target_frame_rate)
    
    print("\nConverting video duration")
    print("Process complete: 25%\n")

    # Process each video file
    video_files = glob.glob(os.path.join(Buffer1, "*.mp4"))
    for video in video_files:
        Buffer = os.path.join(Buffer2, os.path.basename(video))
        adjust_video_length(video, Buffer, target_duration)

    print("\nConverting video to NumPy data")
    print("Process complete: 50%\n")

    # Convert to Numpy data
    numpy_list=[]
    video_files = glob.glob(os.path.join(Buffer2, "*.mp4"))
    for video in video_files:
        data=landmark(video)
        numpy_list.append(data)
    
    print("\nStart prediction")
    print("Process complete: 75%\n")

    # Prediction
    prediction_list=[]
    for data in numpy_list:
        predicted_label = prediction(data)
        label_mapping = {0: 'goodafternoon',1: 'goodmorning',2: 'goodnight',3: 'thankyou'}
        predicted_gesture = label_mapping[predicted_label]
        prediction_list.append(predicted_gesture)
    
    print("\nAnalyze completed")
    print("Process complete: 100%\n")

    # Print the predicted_gesture
    i=1
    for predicted_gesture in prediction_list:
        print(f"The result of Video {i} is {predicted_gesture}")
        i+=1
    

    endTime = time.time()
    runtime = endTime - startTime
    print(f"\nAnalyze time = {runtime}s")