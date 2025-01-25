import os
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import faiss
import open_clip
import csv


features_dir = "" #Ouput of extract_features.py
histogram_dir = ""
map_dir = ""
index_file = ""
metadata_file = ""

device = xm.xla_device()  # Set device to TPU
model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)

# Function to load keyframe mappings from CSV files
def load_keyframe_mappings(map_dir):
    print(f"Loading keyframe mappings from {map_dir}")
    mappings = {}
    for file in os.listdir(map_dir):
        if file.endswith('.csv'):  # Check for .csv files
            try:
                video_id = file.split('.')[0]  # Extract video ID from the filename
                with open(os.path.join(map_dir, file), 'r') as f:
                    csv_reader = csv.DictReader(f)  # Load CSV data into a dictionary
                    mappings[video_id] = [row for row in csv_reader]  # Store mappings for the video
                print(f"Loaded mapping for {video_id}, entries: {len(mappings[video_id])}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")  # Handle exceptions during loading
    print(f"Loaded mappings for {len(mappings)} videos")
    return mappings

# Function to load both visual and histogram features from directories
def load_features_and_histograms(feature_dir, histogram_dir):
    features = {}
    histograms = {}

    for feature_file in os.listdir(feature_dir):
        if feature_file.endswith('.npy'):  # Check for .npy files
            video_id = feature_file.split('.')[0]  # Extract video ID from the filename
            features[video_id] = np.load(os.path.join(feature_dir, feature_file))  # Load numpy array from file

    for histogram_file in os.listdir(histogram_dir):
        if histogram_file.endswith('.npy'):  # Check for .npy files
            video_id = histogram_file.split('.')[0]  # Extract video ID from the filename
            histograms[video_id] = np.load(os.path.join(histogram_dir, histogram_file))  # Load numpy array from file

    return features, histograms

def combine_features(vit_features, hist_features, alpha=0.5):
    return alpha * vit_features + (1 - alpha) * hist_features

# Load features and histograms
features, histograms = load_features_and_histograms(features_dir, histogram_dir)
keyframe_mappings = load_keyframe_mappings(map_dir)

# Prepare data for FAISS
all_combined_features = []
all_metadata = []

for video_id in features.keys():
    video_vit_features = features[video_id]
    video_hist_features = histograms[video_id]
    video_keyframe_mapping = keyframe_mappings[video_id]

    for i in range(len(video_vit_features)):
        combined_feature = combine_features(video_vit_features[i], video_hist_features[i])
        combined_feature = combined_feature / np.linalg.norm(combined_feature)  # Normalize
        all_combined_features.append(combined_feature)
        all_metadata.append((video_id, video_keyframe_mapping[i]))

all_combined_features = np.array(all_combined_features).astype('float32')

# FAISS operations should remain on CPU/GPU
index = faiss.IndexFlatIP(all_combined_features.shape[1])  # Inner Product for cosine similarity
index.add(all_combined_features)

# Save index and metadata
faiss.write_index(index, index_file)
np.save(metadata_file, all_metadata)

print(f"Index and metadata saved to {index_file} and {metadata_file}")