import os
import cv2
import csv

# Define video directory and output directories
video_dir = ''
keyframes_dir = ''
csv_dir = ''

# Create directories if they don't exist
for dir_path in [keyframes_dir, csv_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Function to extract keyframes and generate CSV mapping
def extract_keyframes_and_generate_csv(video_path, video_id, frame_rate=2):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    
    keyframe_dir = os.path.join(keyframes_dir, video_id)
    if not os.path.exists(keyframe_dir):
        os.makedirs(keyframe_dir)
    
    csv_path = os.path.join(csv_dir, f"{video_id}.csv")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'pts_time', 'fps', 'frame_idx'])
        
        success, frame = video_capture.read()
        count = 0
        frame_number = 0
        
        while success:
            if count % interval == 0:
                pts_time = count / fps
                frame_idx = count
                
                # Save keyframe
                keyframe_filename = os.path.join(keyframe_dir, f"{str(frame_number).zfill(4)}.jpg")
                cv2.imwrite(keyframe_filename, frame)
                
                # Write to CSV
                writer.writerow([frame_number + 1, round(pts_time, 2), int(fps), frame_idx])
                frame_number += 1
                
            success, frame = video_capture.read()
            count += 1
    
    video_capture.release()

# Process each video in the directory
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4') and video_file.startswith('L24'):
        video_path = os.path.join(video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        print(f"Processing video: {video_id}")
        
        # Task: Extract keyframes and generate CSV
        extract_keyframes_and_generate_csv(video_path, video_id)

print("Keyframe extraction and CSV mapping completed")