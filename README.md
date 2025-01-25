# Video Retrieval System
- This is CS336 project:
| Member               | Student ID |
|----------------------|------------|
| Trương Quang Lập     | 22520750   |
| Trần Nguyễn Anh Phong| 22521092   |
| Huỳnh Đăng Khoa      | 22520669   |
| Trần Quy Linh        | 22520779   |
## Overview
This project provides a system for video retrieval based on natural language queries. The workflow includes:
1. **Keyframe Extraction**: Extract keyframes from video files.
2. **Feature Extraction**: Extract features from the keyframes using the OpenCLIP model.
3. **FAISS Index Creation**: Build a FAISS index for fast retrieval.
4. **Query and Search**: Retrieve relevant videos based on user queries.
5. **Evaluation**: Measure the system's performance using metrics like Recall@K and Mean Average Precision (MAP).

## File Descriptions

### 1. `create_keyframe.py`
- **Purpose**: Extract keyframes from video files and save their mappings in CSV format.
- **Input**: Videos from a specified directory.
- **Output**: Keyframes saved as images and a CSV file mapping keyframes to timestamps.
- **Usage**:
  ```bash
  python create_keyframe.py
  ```

### 2. `extract_features.py`
- **Purpose**: Load extracted keyframes and compute their visual features using the OpenCLIP model.
- **Input**: Directory containing keyframes.
- **Output**: `.npy` files storing feature vectors for each video.
- **Usage**:
  ```bash
  python extract_features.py
  ```

### 3. `embedding.py`
- **Purpose**: Combine visual and histogram features, build a FAISS index, and save metadata for retrieval.
- **Input**: Feature files and CSV mappings.
- **Output**: FAISS index and metadata files.
- **Usage**:
  ```bash
  python embedding.py
  ```

### 4. `searching.py`
- **Purpose**: Perform search queries on the FAISS index using natural language inputs.
- **Input**: Text query.
- **Output**: List of matching videos with timestamps and similarity scores.
- **Usage**:
  ```bash
  python searching.py
  ```

### 5. `evaluation.py`
- **Purpose**: Evaluate the retrieval system using test data.
- **Input**: FAISS index, metadata, and test dataset.
- **Output**: Metrics such as Recall@K and MAP.
- **Usage**:
  ```bash
  python evaluation.py
  ```

### 6. `test.py`
- **Purpose**: Test the retrieval system with a sample query and output the results.
- **Input**: A sample text query.
- **Output**: Matching videos with details like frame and timestamp.
- **Usage**:
  ```bash
  python test.py
  ```

## Requirements
To install the required dependencies, run:
```bash
pip install -r requirement.txt
```

## Workflow
1. **Extract Keyframes**:
   - Run `create_keyframe.py` to extract keyframes and save their mappings.
2. **Extract Features**:
   - Use `extract_features.py` to generate feature files from keyframes.
3. **Build FAISS Index**:
   - Execute `embedding.py` to create a FAISS index and store metadata.
4. **Search Videos**:
   - Run `searching.py` or `test.py` to query the system.
5. **Evaluate System**:
   - Use `evaluation.py` to compute performance metrics.

## Contact
For any questions or support, please contact [example@example.com].