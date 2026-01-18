import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import pandas as pd

# Load model & preprocessors
print("[INFO] Loading model and preprocessors...")

MODEL_PATH = "src/models_output/MediaPipe/fitness_classifier_randomforest.pkl"
SCALER_PATH = "src/models_output/MediaPipe/scaler.pkl"
ENCODER_PATH = "src/models_output/MediaPipe/label_encoder.pkl"
FEATURE_NAMES_PATH = "src/models_output/MediaPipe/feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
print("[INFO] Model loaded successfully.")

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MP_JOINTS = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "l_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "r_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "l_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "r_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "l_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "r_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "l_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "r_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "l_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "r_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "l_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "r_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# Geometry helpers
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def normalize_skeleton(joints):
    hip_center = (joints["l_hip"] + joints["r_hip"]) / 2
    joints = {k: v - hip_center for k, v in joints.items()}

    torso = np.linalg.norm((joints["l_shoulder"] + joints["r_shoulder"]) / 2)
    torso = max(torso, 1e-6)

    return {k: v / torso for k, v in joints.items()}

# Feature extraction (live)
def extract_features(results):
    if not results.pose_landmarks:
        return None

    joints = {}
    for name, idx in MP_JOINTS.items():
        lm = results.pose_landmarks.landmark[idx]
        joints[name] = np.array([lm.x, lm.y])

    joints = normalize_skeleton(joints)

    features = [
        compute_angle(joints["l_hip"], joints["l_knee"], joints["l_ankle"]),
        compute_angle(joints["r_hip"], joints["r_knee"], joints["r_ankle"]),
        compute_angle(joints["l_shoulder"], joints["l_hip"], joints["l_knee"]),
        compute_angle(joints["r_shoulder"], joints["r_hip"], joints["r_knee"]),
        compute_angle(joints["l_shoulder"], joints["l_elbow"], joints["l_wrist"]),
        compute_angle(joints["r_shoulder"], joints["r_elbow"], joints["r_wrist"]),
    ]

    for k in joints:
        features.extend([joints[k][0], joints[k][1]])

    return np.array(features).reshape(1, -1)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Webcam started. Press 'q' to quit.")

# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARN] Empty frame received, retrying...")
        time.sleep(0.05)
        continue  # IMPORTANT: do NOT break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style()
    )


    features = extract_features(results)
    label_text = "No pose"

    if features is not None:
        features_df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(features_df)
        pred = model.predict(features_scaled)[0]
        label_text = label_encoder.inverse_transform([pred])[0]

    cv2.putText(
        frame,
        f"Exercise: {label_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Exercise Classification (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")
