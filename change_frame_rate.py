###################################################################################################
# THIS IS THE CODE WHICH USED TO CHANGE THE VIDEO FRAME RATE
###################################################################################################
import cv2
import os
import glob

def change_frame_rate(input_video, output_video, target_frame_rate):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
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

if __name__ == "__main__":
    # Example usage
    main_folder = ""
    output_folder = ""
    fps = 30
    os.makedirs(output_folder, exist_ok=True)
    for type_folder in os.listdir(main_folder):
        type_folder_path = os.path.join(main_folder, type_folder)
        if os.path.isdir(type_folder_path):
            output_type_folder = os.path.join(output_folder, type_folder)
            os.makedirs(output_type_folder, exist_ok=True)
            video_files = glob.glob(os.path.join(type_folder_path, "*.mp4"))

            # Process each video file
            for video_file in video_files:
                output_video = os.path.join(output_type_folder, os.path.basename(video_file))
                change_frame_rate(video_file, output_video, fps)
    print("Change complete")

    