import cv2
import numpy as np
import os
import glob

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_video_with_gamma_correction(input_video_path, output_video_path, gamma=2.2):
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
        
        gamma_corrected_frame = adjust_gamma(frame, gamma=gamma)
        out.write(gamma_corrected_frame)
    
    cap.release()
    out.release()

def process_directory_with_gamma_correction(input_dir, output_dir, gamma=2.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))  # Adjust the extension if needed

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_video_path = os.path.join(output_dir, filename)
        
        print(f"Processing {video_path} with gamma correction...")
        process_video_with_gamma_correction(video_path, output_video_path, gamma)
        print(f"Gamma corrected video saved to {output_video_path}")

# Specify your input and output directories here
# Specify your input and output directories here
input_dir = '/content/drive/MyDrive/ThesisProject/Harmonizer/demo/video_enhancement/example/original'  # Update this path
output_dir = '/content/drive/MyDrive/ThesisProject/Harmonizer/demo/video_enhancement/example/gammatranformed'  # Update this path

# Set the gamma value for correction
gamma_value = 0.5  # Change this value as needed

process_directory_with_gamma_correction(input_dir, output_dir, gamma=gamma_value)
