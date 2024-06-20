###################################################################################################
# THIS IS THE CODE WHICH USED TO MASSIVE CONVERT VIDEO INTO NUMPY DATA
###################################################################################################
import os
import glob
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# Function to process a video and generate numpy files
def process_video(video_path, output_folder_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create a folder to save numpy files for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder_path, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    numpy_file_count = 0  # Counter for naming the numpy files

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Only process frames at the desired interval
        else:
            # Convert the frame to PIL Image format
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_data = encoder(frame_pil)
            numpy_file_path = os.path.join(video_output_folder, f"{numpy_file_count}.npy")
            np.save(numpy_file_path, frame_data)

            numpy_file_count += 1  # Increment the numpy file counter

    video.release()

if __name__ == "__main__":
    main_folder = ""
    output_folder = ""

    os.makedirs(output_folder, exist_ok=True)
    start_time = time.time()
    # Use multiprocessing to parallelize video processing across different processes
    with ProcessPoolExecutor() as process_executor:
        # Iterate through the main folder and its type-folders
        for type_folder in os.listdir(main_folder):
            type_folder_path = os.path.join(main_folder, type_folder)
            if os.path.isdir(type_folder_path):
                output_type_folder = os.path.join(output_folder, type_folder)
                os.makedirs(output_type_folder, exist_ok=True)
                video_files = glob.glob(os.path.join(type_folder_path, "*.mp4"))

                # Use multithreading to parallelize video processing within each process
                with ThreadPoolExecutor() as thread_executor:
                    # Process each video file in parallel using multithreading
                    thread_executor.map(process_video, video_files, [output_type_folder] * len(video_files))

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")