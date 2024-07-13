###################################################################################################
# THIS IS THE CODE WHICH USED TO CHANGE THE VIDEO LENGTH
###################################################################################################
from moviepy.editor import VideoFileClip, concatenate, vfx
import os
import glob
def extend_video(video, target_duration, output_folder):
    clip = VideoFileClip(video)
    extension_clip = clip.subclip(clip.duration - target_duration)
    extended_clip = concatenate([clip, extension_clip])
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video))[0] + ".mp4")
    extended_clip.write_videofile(output_path)
    clip.close()
    extension_clip.close()
    extended_clip.close()

def compress_video(video, target_duration, output_folder):
    clip = VideoFileClip(video)
    clip_duration = clip.duration
    compression_factor = clip_duration / target_duration
    compressed_clip = clip.fx(vfx.speedx, compression_factor)
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video))[0] + ".mp4")
    compressed_clip.write_videofile(output_path)
    clip.close()
    compressed_clip.close()
    
def adjust_video_length(video, output_folder, target_duration):
    clip = VideoFileClip(video)
    duration = clip.duration

    if duration < target_duration:
        extend_video(video, target_duration, output_folder)
    elif duration > target_duration:
        compress_video(video, target_duration, output_folder)

    clip.close()

if __name__ == "__main__":
    # Example usage
    main_folder = ""
    output_folder = ""
    target_duration = 1  # seconds
    os.makedirs(output_folder, exist_ok=True)
    for type_folder in os.listdir(main_folder):
        type_folder_path = os.path.join(main_folder, type_folder)
        if os.path.isdir(type_folder_path):
            output_type_folder = os.path.join(output_folder, type_folder)
            os.makedirs(output_type_folder, exist_ok=True)
            video_files = glob.glob(os.path.join(type_folder_path, "*.mp4"))
    
            # Process each video file
            for video_file in video_files:
                adjust_video_length(video_file, output_type_folder, target_duration)
    print("Change complete")