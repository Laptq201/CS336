import json
import numpy as np
import faiss
import open_clip
import torch
from collections import defaultdict

# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)

# Load FAISS index and metadata
index_file = "/home/lapquang/Downloads/archive/MSR-VTT/combined_features.index"
metadata_file = "/home/lapquang/Downloads/archive/MSR-VTT/metadata.npy"
index = faiss.read_index(index_file)
all_metadata = np.load(metadata_file, allow_pickle=True)


import time

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
    
query = "A person is running in a green field"

# Perform the search
start = time.time()
results = search_index(query, index, all_metadata, clip_model, tokenizer, k=10)
end = time.time()-start
# Print results
print(end)
for video_id, frame_number, timestamp, similarity in results:
    print(f"Video ID: {video_id}, Frame: {frame_number}, Time: {timestamp}s, Similarity: {similarity}")

# Evaluation metrics
#recall_at_k = {1: 0, 5: 0, 10: 0}
#total_queries = 0

# Perform evaluation
#for video_id, captions in video_to_caption.items():
#    for caption in captions:
#        total_queries += 1
#        ranked_results = search_index(caption, index, all_metadata, clip_model, tokenizer)
#
#        retrieved_video_ids = [result[0] for result in ranked_results]

        # Check if the correct video is within the top-k results
#        for k in recall_at_k.keys():
#            if video_id in retrieved_video_ids[:k]:
#                recall_at_k[k] += 1