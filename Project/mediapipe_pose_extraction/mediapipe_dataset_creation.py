import cv2
import pandas as pd
from tqdm import tqdm
import os
import mediapipe as mp

# Inside container, these paths will be volumes
DATA_ROOT = "../data/"
OUTPUT_CSV = "/app/output/mediapipe_pose_dataset.csv"


MP_JOINTS = {
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "l_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "r_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "l_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "r_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "l_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "r_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "l_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "r_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "l_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "r_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "l_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "r_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}

import numpy as np

def compute_angle(a, b, c):
    """Angle at point b (in degrees)"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def normalize_skeleton(joints):
    """
    joints: dict {name: np.array([x, y])}
    """
    # Center at hip midpoint
    hip_center = (joints["l_hip"] + joints["r_hip"]) / 2
    joints = {k: v - hip_center for k, v in joints.items()}

    # Scale by torso length
    torso = np.linalg.norm(
        ((joints["l_shoulder"] + joints["r_shoulder"]) / 2)
    )
    torso = max(torso, 1e-6)

    joints = {k: v / torso for k, v in joints.items()}
    return joints



mp_pose = mp.solutions.pose

def process_image(image_path, pose_model):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = pose_model.process(image_rgb)
    if not result.pose_landmarks:
        return None

    joints = {}
    confidences = []

    for name, idx in MP_JOINTS.items():
        lm = result.pose_landmarks.landmark[idx]
        joints[name] = np.array([lm.x, lm.y])
        confidences.append(lm.visibility)

    # Confidence filtering
    if np.mean(confidences) < 0.5:
        return None

    joints = normalize_skeleton(joints)

    # Angles
    features = {}

    features["knee_angle_l"] = compute_angle(
        joints["l_hip"], joints["l_knee"], joints["l_ankle"]
    )
    features["knee_angle_r"] = compute_angle(
        joints["r_hip"], joints["r_knee"], joints["r_ankle"]
    )
    features["hip_angle_l"] = compute_angle(
        joints["l_shoulder"], joints["l_hip"], joints["l_knee"]
    )
    features["hip_angle_r"] = compute_angle(
        joints["r_shoulder"], joints["r_hip"], joints["r_knee"]
    )
    features["elbow_angle_l"] = compute_angle(
        joints["l_shoulder"], joints["l_elbow"], joints["l_wrist"]
    )
    features["elbow_angle_r"] = compute_angle(
        joints["r_shoulder"], joints["r_elbow"], joints["r_wrist"]
    )

    # Flatten joint coordinates
    for k, v in joints.items():
        features[f"{k}_x"] = v[0]
        features[f"{k}_y"] = v[1]

    return features

DATASET_ROOT = "./data/"

rows = []

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
) as pose:

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
            features["image"] = img_name
            rows.append(features)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)
print("Final dataset shape:", df.shape)
