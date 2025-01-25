import json
import numpy as np
import faiss
import open_clip
import torch
from collections import defaultdict


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)

# Load FAISS index and metadata
index_file = "/kaggle/input/msrvtt/MSR-VTT/combined_features.index"
metadata_file = "/kaggle/input/msrvtt/MSR-VTT/metadata.npy"
index = faiss.read_index(index_file)
all_metadata = np.load(metadata_file, allow_pickle=True)


with open("/kaggle/input/msrvtt/MSR-VTT/test_videodatainfo.json") as f:
    test_data = json.load(f)

# Create a mapping from video ID to captions
video_to_caption = defaultdict(list)
for sentence in test_data['sentences']:
    video_to_caption[sentence['video_id']].append(sentence['caption'])


def encode_text_query(query, model, tokenizer):
    text = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).squeeze().cpu().numpy()
    return text_features

def search_index(query, index, metadata, clip_model, tokenizer, k=10):
    text_features = encode_text_query(query, clip_model, tokenizer)
    text_features = np.array([text_features]).astype('float32')
    text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)  # Normalize text features

    # Perform the search
    distances, indices = index.search(text_features, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        video_id, frame_info = metadata[idx]
        frame_number = int(frame_info['frame_idx'])
        timestamp = float(frame_info['pts_time'])
        results.append((video_id, frame_number, timestamp, dist))

    results.sort(key=lambda x: x[3], reverse=True)  # Sort by distance (higher is better for cosine similarity)
    return results

# Evaluation metrics
