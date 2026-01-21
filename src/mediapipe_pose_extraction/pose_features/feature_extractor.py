import numpy as np
from .geometry import compute_angle, compute_inclination
from .normalization import normalize_skeleton

def extract_features(joints):
    shoulder_mid = (joints["l_shoulder"] + joints["r_shoulder"]) / 2
    hip_mid = (joints["l_hip"] + joints["r_hip"]) / 2
    
    torso_inclination = compute_inclination(shoulder_mid, hip_mid)

    joints_norm = normalize_skeleton(joints)

    features = {}

    features["torso_inclination"] = torso_inclination

    features["knee_angle_l"] = compute_angle(
        joints_norm["l_hip"], joints_norm["l_knee"], joints_norm["l_ankle"]
    )
    features["knee_angle_r"] = compute_angle(
        joints_norm["r_hip"], joints_norm["r_knee"], joints_norm["r_ankle"]
    )
    features["hip_angle_l"] = compute_angle(
        joints_norm["l_shoulder"], joints_norm["l_hip"], joints_norm["l_knee"]
    )
    features["hip_angle_r"] = compute_angle(
        joints_norm["r_shoulder"], joints_norm["r_hip"], joints_norm["r_knee"]
    )
    features["elbow_angle_l"] = compute_angle(
        joints_norm["l_shoulder"], joints_norm["l_elbow"], joints_norm["l_wrist"]
    )
    features["elbow_angle_r"] = compute_angle(
        joints_norm["r_shoulder"], joints_norm["r_elbow"], joints_norm["r_wrist"]
    )

    return features