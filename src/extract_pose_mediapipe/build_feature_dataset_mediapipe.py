from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

NPZ_DIR = Path("../output/mediapipe_pose_npz")              # Input: pro Video .npz mit X (T,33,4)
OUT_CSV = Path("../data/mediapipe_dataset.csv")  # Output: tabellarische Features pro Window

FPS_TARGET = 15
WIN_SEC = 2.0
STEP_SEC = 1.0

WIN = int(WIN_SEC * FPS_TARGET)   # z.B. 30
STEP = int(STEP_SEC * FPS_TARGET) # z.B. 15

VIS_THRESH = 0.5          # Frame gilt als "valid", wenn mean visibility >= threshold
MIN_VALID_FRAC = 0.6      # Window wird nur genutzt, wenn genug valide Frames drin sind

# MediaPipe Pose landmark indices (relevant subset)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

def _angle(a, b, c):
    """
    Winkel in Grad am Punkt b, gebildet von (a-b) und (c-b).
    a,b,c: (3,) arrays
    """
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0 or np.any(np.isnan([nba, nbc])):
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def _dist(a, b):
    d = np.linalg.norm(a - b)
    return d if not np.isnan(d) else np.nan

def per_frame_features(X):
    """
    X: (T,33,4) -> returns:
      feats: dict[name] -> (T,) float
      valid: (T,) bool
      scale: (T,) float (torso scale)
    """
    T = X.shape[0]
    xyz = X[:, :, :3]          # (T,33,3)
    vis = X[:, :, 3]           # (T,33)

    # Keypoints, die wir für validity checken
    key_idx = np.array([L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ELBOW, R_ELBOW])
    mean_vis = np.nanmean(vis[:, key_idx], axis=1)
    valid = mean_vis >= VIS_THRESH

    # Hilfspunkte
    shoulder_mid = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])  # (T,3)
    hip_mid      = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])            # (T,3)

    shoulder_width = np.array([_dist(xyz[t, L_SHOULDER], xyz[t, R_SHOULDER]) for t in range(T)], dtype=np.float32)
    hip_width      = np.array([_dist(xyz[t, L_HIP], xyz[t, R_HIP]) for t in range(T)], dtype=np.float32)
    torso_len      = np.array([_dist(shoulder_mid[t], hip_mid[t]) for t in range(T)], dtype=np.float32)

    # Skala: robust, vermeide 0
    scale = np.nan_to_num(torso_len, nan=0.0)
    scale = np.where(scale < 1e-6, np.nan, scale)

    def norm_dist(p, q):
        d = np.array([_dist(p[t], q[t]) for t in range(T)], dtype=np.float32)
        return d / scale

    # Winkel (Grad)
    elbow_l = np.array([_angle(xyz[t, L_SHOULDER], xyz[t, L_ELBOW], xyz[t, L_WRIST]) for t in range(T)], dtype=np.float32)
    elbow_r = np.array([_angle(xyz[t, R_SHOULDER], xyz[t, R_ELBOW], xyz[t, R_WRIST]) for t in range(T)], dtype=np.float32)

    knee_l  = np.array([_angle(xyz[t, L_HIP], xyz[t, L_KNEE], xyz[t, L_ANKLE]) for t in range(T)], dtype=np.float32)
    knee_r  = np.array([_angle(xyz[t, R_HIP], xyz[t, R_KNEE], xyz[t, R_ANKLE]) for t in range(T)], dtype=np.float32)

    hip_l   = np.array([_angle(xyz[t, L_SHOULDER], xyz[t, L_HIP], xyz[t, L_KNEE]) for t in range(T)], dtype=np.float32)
    hip_r   = np.array([_angle(xyz[t, R_SHOULDER], xyz[t, R_HIP], xyz[t, R_KNEE]) for t in range(T)], dtype=np.float32)

    shoulder_l = np.array([_angle(xyz[t, L_ELBOW], xyz[t, L_SHOULDER], xyz[t, L_HIP]) for t in range(T)], dtype=np.float32)
    shoulder_r = np.array([_angle(xyz[t, R_ELBOW], xyz[t, R_SHOULDER], xyz[t, R_HIP]) for t in range(T)], dtype=np.float32)

    # Distanzen (normalisiert auf torso_len)
    wrist2hip_l = norm_dist(xyz[:, L_WRIST], hip_mid)
    wrist2hip_r = norm_dist(xyz[:, R_WRIST], hip_mid)
    ankle2hip_l = norm_dist(xyz[:, L_ANKLE], hip_mid)
    ankle2hip_r = norm_dist(xyz[:, R_ANKLE], hip_mid)

    feats = {
        "mean_vis": mean_vis.astype(np.float32),
        "shoulder_width": shoulder_width.astype(np.float32),
        "hip_width": hip_width.astype(np.float32),
        "torso_len": torso_len.astype(np.float32),

        "elbow_l": elbow_l, "elbow_r": elbow_r,
        "knee_l": knee_l, "knee_r": knee_r,
        "hip_l": hip_l, "hip_r": hip_r,
        "shoulder_l": shoulder_l, "shoulder_r": shoulder_r,

        "wrist2hip_l": wrist2hip_l.astype(np.float32),
        "wrist2hip_r": wrist2hip_r.astype(np.float32),
        "ankle2hip_l": ankle2hip_l.astype(np.float32),
        "ankle2hip_r": ankle2hip_r.astype(np.float32),
    }

    # Ungültige Frames -> NaN setzen (damit aggregation sauber ist)
    for k in list(feats.keys()):
        v = feats[k].copy()
        v[~valid] = np.nan
        feats[k] = v

    return feats, valid

def agg_window(vec):
    """vec: (WIN,) with NaNs -> returns dict of aggregations"""
    out = {}
    out["mean"] = float(np.nanmean(vec))
    out["std"]  = float(np.nanstd(vec))
    out["min"]  = float(np.nanmin(vec))
    out["max"]  = float(np.nanmax(vec))

    # simple motion: mean absolute diff
    if np.sum(~np.isnan(vec)) >= 2:
        dv = np.diff(vec)
        out["madiff"] = float(np.nanmean(np.abs(dv)))
    else:
        out["madiff"] = np.nan
    return out

def main():
    rows = []
    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No npz files found in {NPZ_DIR}. Did you run extract_pose.py?")

    for npz_path in tqdm(npz_files, desc="Building features"):
        d = np.load(npz_path, allow_pickle=True)
        X = d["X_img"]
        label = str(d["label"]) if "label" in d else npz_path.name.split("__")[0]
        video_path = str(d["video_path"]) if "video_path" in d else ""

        if X.shape[0] < WIN:
            continue

        feats, valid = per_frame_features(X)
        T = X.shape[0]

        for start in range(0, T - WIN + 1, STEP):
            end = start + WIN
            valid_frac = float(np.mean(valid[start:end]))
            if valid_frac < MIN_VALID_FRAC:
                continue

            row = {
                "label": label,
                "npz_source": str(npz_path),
                "video_path": video_path,
                "start_frame": start,
                "end_frame": end,
                "valid_frac": valid_frac,
            }

            # Feature-Aggregation pro Window
            for name, series in feats.items():
                w = series[start:end]
                # komplett NaN -> skip
                if np.all(np.isnan(w)):
                    continue
                a = agg_window(w)
                for stat, val in a.items():
                    row[f"{name}__{stat}"] = val

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)
    print("Shape:", df.shape)
    print("Classes:", df["label"].nunique() if "label" in df else "n/a")

if __name__ == "__main__":
    main()