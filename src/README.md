# Versions
Python: 3.13

# Setup Project
## 1. Install Requirements
```bash
pip install -r requirements.txt
```

## 2. Download the Data
1. Download the dataset from [this link](https://drive.google.com/file/d/1fC0vEjUotKBJ37qbYx7WfWDjyYAJU718/view?usp=share_link)
2. Unzip the downloaded file. You should see a folder named `data`.
3. Replace the existing `data` folder in this repository with the newly unzipped `data` folder.
4. It should contain one folder per exercise

## 3. Download the MediaPipe Models
Download "pose_landmarker_heavy.task" from [this link](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) and place it in 'extract_pose_mediapipe' folder.
[Alternative Link](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=de#models)

# Run the Project
## 1. Create the Manifest
Run the "make_manifest.py" script to create a manifest of the dataset.
```bash
python extract_pose_mediapipe/make_manifest.py
```

## 2. Extract Pose Data
Run the "extract_pose.py" script to extract pose data from the videos.
```bash
python extract_pose_mediapipe/extract_pose.py
```

## 3. Build the Feature Dataset
Run the "build_feature_dataset.py" script to build the feature dataset.
```bash
python extract_pose_mediapipe/build_feature_dataset.py
```

## 4. Train the Classification Models
Run the "train.py" script to train the classification models.
```bash
python models/train.py
```

## 5. Show live webcam demo
Run the "demo_mediapipe_macos.py" script to see a live webcam demo.
```bash
python demo/webcam_demo.py
```

