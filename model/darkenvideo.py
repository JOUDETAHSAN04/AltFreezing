import cv2
import numpy as np
import os
import glob

def darken_video(input_video_path, output_video_path, brightness_factor=0.5):
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        dark_frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
        out.write(dark_frame)
    
    cap.release()
    out.release()

def process_directory(input_dir, output_dir, brightness_factor=0.5):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all video files in the input directory
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))  # Adjust the extension if needed

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_video_path = os.path.join(output_dir, filename)
        
        print(f"Processing {video_path}...")
        darken_video(video_path, output_video_path, brightness_factor)
        print(f"Darkened video saved to {output_video_path}")

# Specify your input and output directories here
input_dir = '/content/drive/MyDrive/ThesisProject/Harmonizer/demo/video_enhancement/example/original'  # Update this path
output_dir = '/content/drive/MyDrive/ThesisProject/Harmonizer/demo/video_enhancement/example/darkened'  # Update this path

process_directory(input_dir, output_dir, brightness_factor=0.5)
