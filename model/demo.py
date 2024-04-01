import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader


mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

def process_video(video_path, out_dir, cfg_path, ckpt_path, optimal_threshold, max_frame=400):
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load(ckpt_path)

    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    os.makedirs(out_dir, exist_ok=True)
    basename = f"{os.path.splitext(os.path.basename(video_path))[0]}.avi"
    out_file = os.path.join(out_dir, basename)

    cache_file = f"{video_path}_{max_frame}.pth"

    prediction_values = []
    frame_id_list = []

    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        print("Detection result loaded from cache.")
    else:
        print("Detecting...")
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)
        print("Detection finished.")

        print("Number of frames: ", len(frames))

    shape = frames[0].shape[:2]
    all_detect_res = []
    assert len(all_lm68) == len(detect_res)

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)

    detect_res = all_detect_res
    print("Split into super clips")
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    print("Full_tracks", len(tracks))
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        assert len(detect_res[start:end]) == len(track)
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)
            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(int)

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        # Include padding logic here if super_clip_size < clip_size
        # Adapted for brevity

        frame_range = [inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    preds = []
    frame_res = {}

    for clip in tqdm(clips_for_video, desc="Testing clips"):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        _, images_align = crop_align_func(landmarks, images)
      
        images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images)
        pred = float(F.sigmoid(output["final_output"]))
        preds.append(pred)

        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)

    # Aggregate predictions and prepare for saving
    df = pd.DataFrame({'Frame ID': list(frame_res.keys()), 'Prediction': [np.mean(frame_res[f_id]) for f_id in frame_res.keys()]})
    excel_file_path = os.path.join(out_dir, f"predictions_{os.path.basename(video_path)}.xlsx")
    df.to_excel(excel_file_path, index=False)

    print(f"Processed {len(frames)} frames. Mean prediction: {np.mean(preds)}")

        # Assuming all_detect_res preparation and prediction logic have been completed above

    # Process each frame and predict
    preds = []
    frame_res = {}

    # Loop through your structured data to make predictions
    for clip in tqdm(clips_for_video, desc="Testing clips"):
        # Adjust gamma if necessary, preprocess, and collect frames for prediction
        #gamma_value = 0.09
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        
        _, images_align = crop_align_func(landmarks, images)
        images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images)
        pred = float(torch.sigmoid(output["final_output"]))
        preds.append(pred)
        
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)

    # Calculate the average prediction for frames involved in multiple predictions
    for frame_id in frame_res:
        frame_res[frame_id] = np.mean(frame_res[frame_id])

    # Create a list of all frame indices
    all_frame_ids = list(range(len(frames)))

    # Prepare scores and boxes for SupplyWriter
    scores = [frame_res.get(frame_id, None) for frame_id in all_frame_ids]
    boxes = [frame_boxes.get(frame_id, None) for frame_id in all_frame_ids]

    # Use SupplyWriter or a similar utility to handle the output based on the predictions
    SupplyWriter(video_path, out_file, optimal_threshold).run(frames, scores, boxes)

    print(f"Output video has been saved to: {out_file}")



    print("Processing complete for video:", video_path)
    predictions = [np.mean(frame_res[f_id]) if f_id in frame_res else None for f_id in sorted(frame_res)]
    frame_ids = sorted(frame_res)
    df = pd.DataFrame({'Frame ID': frame_ids, video_path: predictions})
    return df

if __name__ == "__main__":
    video_dir = "/content/drive/MyDrive/ThesisProject/Harmonizer/demo/video_enhancement/example/original"
    out_dir = "predictions-OG"
    cfg_path = "i3d_ori.yaml"
    ckpt_path = "checkpoints/model.pth"
    optimal_threshold = 0.04

    combined_df = pd.DataFrame()

    for video_file in glob.glob(os.path.join(video_dir, '*.mp4')):
        print(f"Processing video: {video_file}")
        video_df = process_video(video_file, out_dir, cfg_path, ckpt_path, optimal_threshold, 400)
        
        # If it's the first video, initialize combined_df with this DataFrame
        if combined_df.empty:
            combined_df = video_df.set_index('Frame ID')
        else:
            combined_df = combined_df.merge(video_df.set_index('Frame ID'), left_index=True, right_index=True, how='outer')

    # After processing all videos, reset the index for better Excel formatting
    combined_df.reset_index(inplace=True)

    # Saving the combined DataFrame to an Excel file
    excel_file_path = os.path.join(out_dir, "combined_video_predictionsOG.xlsx")
    combined_df.to_excel(excel_file_path, index=False)
