import numpy as np

def normalize_skeleton(joints):
    hip_center = (joints["l_hip"] + joints["r_hip"]) / 2
    joints = {k: v - hip_center for k, v in joints.items()}

    torso = np.linalg.norm(
        (joints["l_shoulder"] + joints["r_shoulder"]) / 2
    )
    torso = max(torso, 1e-6)

    joints = {k: v / torso for k, v in joints.items()}
    return joints
