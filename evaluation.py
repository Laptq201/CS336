import json
import numpy as np
import faiss
import open_clip
import torch
from collections import defaultdict
from searching import encode_text_query, search_index


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

def calculate_map(video_to_caption, index, all_metadata, clip_model, tokenizer, k=10):
    average_precisions = []

    for video_id, captions in video_to_caption.items():
        for caption in captions:
            # Get ranked results for the query
            ranked_results = search_index(caption, index, all_metadata, clip_model, tokenizer, k=k)
            retrieved_video_ids = [result[0] for result in ranked_results]

            # Compute Average Precision (AP) for the query
            num_relevant = 0
            precision_at_k = []
            
            for i, retrieved_video_id in enumerate(retrieved_video_ids, start=1):
                if retrieved_video_id == video_id:  # If the retrieved video is relevant
                    num_relevant += 1
                    precision_at_k.append(num_relevant / i)  # Precision at rank i
            
            if num_relevant > 0:
                average_precision = sum(precision_at_k) / num_relevant
                average_precisions.append(average_precision)
            else:
                average_precisions.append(0)  # No relevant results for this query

    # Calculate Mean Average Precision (MAP)
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0
    return map_score


recall_at_k = {1: 0, 5: 0, 10: 0}
total_queries = 0

# Perform evaluation
for video_id, captions in video_to_caption.items():
    for caption in captions:
        total_queries += 1
        ranked_results = search_index(caption, index, all_metadata, clip_model, tokenizer)

        retrieved_video_ids = [result[0] for result in ranked_results]

        # Check if the correct video is within the top-k results
        for k in recall_at_k.keys():
            if video_id in retrieved_video_ids[:k]:
                recall_at_k[k] += 1

# Calculate Recall@K
for k in recall_at_k.keys():
    recall_at_k[k] /= total_queries
    print(f"Recall@{k}: {recall_at_k[k]:.4f}")

map_score = calculate_map(video_to_caption, index, all_metadata, clip_model, tokenizer, k=10)
print(f"Mean Average Precision (MAP@10): {map_score:.4f}")