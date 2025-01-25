import os
import torch
import open_clip
import numpy as np
from PIL import Image
import torch_xla.core.xla_model as xm
import cv2

keyframes_dir = ""
output_dir = ""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


os.makedirs(output_dir, exist_ok=True)

def load_keyframes(video_id):
    """
    Load keyframes from the directory corresponding to the video_id.
    """
    frames_dir = os.path.join(keyframes_dir, video_id)
    print(frames_dir)
    keyframes = []
    for frame_file in sorted(os.listdir(frames_dir)):
        print(frame_file)
        if frame_file.endswith('.jpg'):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            keyframes.append(frame)
    return keyframes

# Function to preprocess keyframes
def preprocess_frames(keyframes):
    preprocessed_frames = []
    for keyframe in keyframes:
        image = Image.fromarray(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB))  # Corrected variable name
        preprocessed_image = preprocess_val(image).unsqueeze(0).to(device)
        preprocessed_frames.append(preprocessed_image)
    return preprocessed_frames

# Function to process frames in batches
def process_frames_in_batches(frames, batch_size=16):
    features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_frames_tensor = torch.cat(batch_frames)
        with torch.no_grad():
            batch_features = model.encode_image(batch_frames_tensor)
            features.append(batch_features.cpu().numpy())
        # Release memory
        del batch_frames_tensor
        xm.mark_step()
    return np.concatenate(features, axis=0)

# Load OpenCLIP model
device = xm.xla_device()
print(f"Using device: {device}")
model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
model, preprocess_val, preprocess_train = open_clip.create_model_and_transforms(model_name)
model = model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)
print("OpenCLIP model loaded")

video_dirs = [f for f in os.listdir(keyframes_dir) 
              if f.startswith('L30') and 
              os.path.isdir(os.path.join(keyframes_dir, f))]

for video_dir in video_dirs:
    video_id = video_dir
    print(f"Processing keyframes for video: {video_id}")
    try:
        frames = load_keyframes(video_id)
        preprocessed_frames = preprocess_frames(frames)
        features = process_frames_in_batches(preprocessed_frames)
        output_path = os.path.join(output_dir, f"{video_id}.npy")
        np.save(output_path, features)
        print(f"Saved features for {video_id} to {output_path}")
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")