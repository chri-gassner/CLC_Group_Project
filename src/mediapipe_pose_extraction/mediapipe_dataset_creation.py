import cv2
import pandas as pd
from tqdm import tqdm
import os
import mediapipe as mp

from pose_features.mediapipe_adapter import extract_joints_from_result
from pose_features.feature_extractor import extract_features


# Inside container, these paths will be volumes
DATA_ROOT = "../data/"
OUTPUT_CSV = "/app/output/mediapipe_pose_dataset.csv"

import numpy as np

mp_pose = mp.solutions.pose

def process_image(image_path, pose_model):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = pose_model.process(image_rgb)
    if not result.pose_landmarks:
        return None

    joints, confidences = extract_joints_from_result(result)

    if joints is None:
        return None
    
    # Confidence filtering
    if np.mean(confidences) < 0.5:
        return None

    features = extract_features(joints)

    return features

DATASET_ROOT = "./data/"

rows = []

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
) as pose:
    if os.path.exists(DATA_ROOT):
        for exercise in os.listdir(DATASET_ROOT):
            exercise_path = os.path.join(DATASET_ROOT, exercise)
            if not os.path.isdir(exercise_path):
                continue

            for img_name in tqdm(os.listdir(exercise_path), desc=exercise):
                img_path = os.path.join(exercise_path, img_name)

                features = process_image(img_path, pose)
                if features is None:
                    continue

                features["label"] = exercise
                features["file"] = img_name
                rows.append(features)
    else:
        print(f"Data root path '{DATA_ROOT}' does not exist.")

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)
print("Final dataset shape:", df.shape)
