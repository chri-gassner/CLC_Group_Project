from .geometry import compute_angle
from .normalization import normalize_skeleton

def extract_features(joints):
    joints = normalize_skeleton(joints)

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

    for k, v in joints.items():
        features[f"{k}_x"] = v[0]
        features[f"{k}_y"] = v[1]

    return features
