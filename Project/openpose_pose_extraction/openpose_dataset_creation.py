import numpy as np
import os
import pandas as pd
from tqdm import tqdm

OPENPOSE_ROOT = "/app/openpose/json"
OUTPUT_CSV = "/app/output/openpose_fitness_dataset.csv"

OP_JOINTS = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 2,
    "l_elbow": 6,
    "r_elbow": 3,
    "l_wrist": 7,
    "r_wrist": 4,
    "l_hip": 11,
    "r_hip": 8,
    "l_knee": 12,
    "r_knee": 9,
    "l_ankle": 13,
    "r_ankle": 10,
}

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def normalize_skeleton(joints):
    hip_center = (joints["l_hip"] + joints["r_hip"]) / 2
    joints = {k: v - hip_center for k, v in joints.items()}

    torso = np.linalg.norm(
        (joints["l_shoulder"] + joints["r_shoulder"]) / 2
    )
    torso = max(torso, 1e-6)

    joints = {k: v / torso for k, v in joints.items()}
    return joints

import json

def process_openpose_json(json_path, conf_thresh=0.5):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "people" not in data or len(data["people"]) == 0:
        return None


    keypoints = data["people"][0]["pose_keypoints_2d"]

    joints = {}
    confidences = []

    for name, idx in OP_JOINTS.items():
        x = keypoints[3 * idx]
        y = keypoints[3 * idx + 1]
        c = keypoints[3 * idx + 2]

        joints[name] = np.array([x, y])
        confidences.append(c)

    # Confidence filtering
    if np.mean(confidences) < conf_thresh:
        return None

    joints = normalize_skeleton(joints)

    features = {}

    # Angles (same as MediaPipe)
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

    # Flatten joints
    for k, v in joints.items():
        features[f"{k}_x"] = v[0]
        features[f"{k}_y"] = v[1]

    return features


rows = []

for exercise in os.listdir(OPENPOSE_ROOT):
    exercise_path = os.path.join(OPENPOSE_ROOT, exercise)
    if not os.path.isdir(exercise_path):
        continue

    for fname in tqdm(os.listdir(exercise_path), desc=exercise):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(exercise_path, fname)
        features = process_openpose_json(json_path)

        if features is None:
            continue

        features["label"] = exercise
        features["file"] = fname
        rows.append(features)


df_op = pd.DataFrame(rows)
df_op.to_csv(OUTPUT_CSV, index=False)

print("OpenPose dataset shape:", df_op.shape)
print("Saved to:", OUTPUT_CSV)
