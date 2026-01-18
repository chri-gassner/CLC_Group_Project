import numpy as np
import mediapipe as mp

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

def extract_joints_from_result(result):
    if not result.pose_landmarks:
        return None, None

    joints = {}
    confidences = []

    for name, idx in MP_JOINTS.items():
        lm = result.pose_landmarks.landmark[idx]
        joints[name] = np.array([lm.x, lm.y])
        confidences.append(lm.visibility)

    return joints, confidences
