#!/usr/bin/env python3
"""
build_windows_openpose.py

Build tabular window-features from an OpenPose per-frame CSV (output of extract_pose_openpose.py).

Input:  per-frame CSV with columns like:
  - label
  - file (OpenPose json filename, e.g. video1_000000000123_keypoints.json)
  - mean_conf, valid_joints (if you used my extract_pose_openpose.py)
  - numeric feature columns (angles, joint_x/y, joint_c, ...)

Output: one row per sliding window with aggregated stats (mean/std/min/max/madiff).

Notes:
- OpenPose JSON doesn't carry FPS. We assume FPS_ASSUMED for seconds->frames conversion.
- Optional downsample to FPS_TARGET by stride computed from FPS_ASSUMED.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# CONFIG (edit here)
# =========================
IN_CSV = Path("../output/openpose_fitness_dataset.csv")   # <-- your per-frame CSV
OUT_CSV = Path("../data/openpose_dataset_windows.csv")

# Windowing setup (same style as your MediaPipe file)
FPS_TARGET = 15
FPS_ASSUMED = 30.0          # OpenPose JSON doesn't store fps -> assume
WIN_SEC = 2.0
STEP_SEC = 1.0

WIN = int(WIN_SEC * FPS_TARGET)     # e.g. 30
STEP = int(STEP_SEC * FPS_TARGET)   # e.g. 15

# Validity filtering (uses columns if present)
CONF_THRESH_MEAN = 0.5      # requires mean_conf >= this if column exists
MIN_VALID_JOINTS = 6        # requires valid_joints >= this if column exists
MIN_VALID_FRAC = 0.6        # window kept only if valid_frac >= this

# How to group frames into one "sequence" (= one video)
# We derive video_id from the JSON filename by stripping frame index suffix.
# Example: video1_000000000123_keypoints.json -> video1
VIDEO_ID_REGEX = re.compile(r"^(.*?)(?:_\d+)?_keypoints\.json$", re.IGNORECASE)

# Frame index extraction
FRAME_IDX_REGEX = re.compile(r"_(\d+)_keypoints\.json$", re.IGNORECASE)


# =========================
# Helpers
# =========================
def parse_video_id(fname: str) -> str:
    m = VIDEO_ID_REGEX.match(fname)
    return m.group(1) if m else Path(fname).stem

def parse_frame_idx(fname: str) -> int:
    m = FRAME_IDX_REGEX.search(fname)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1

def agg_window(vec: np.ndarray) -> dict:
    """vec: (WIN,) float with NaNs -> dict of aggregations"""
    vec = np.asarray(vec, dtype=np.float32)

    out = {
        "mean": float(np.nanmean(vec)) if np.any(np.isfinite(vec)) else np.nan,
        "std":  float(np.nanstd(vec))  if np.any(np.isfinite(vec)) else np.nan,
        "min":  float(np.nanmin(vec))  if np.any(np.isfinite(vec)) else np.nan,
        "max":  float(np.nanmax(vec))  if np.any(np.isfinite(vec)) else np.nan,
    }

    finite = vec[np.isfinite(vec)]
    if finite.size >= 2:
        dv = np.diff(vec)
        out["madiff"] = float(np.nanmean(np.abs(dv)))
    else:
        out["madiff"] = np.nan

    return out

def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    # Exclude known meta columns; keep numeric
    meta = {
        "label", "file", "json_path",
        "video_path", "npz_source",
        "start_frame", "end_frame", "valid_frac",
        "mean_conf", "valid_joints"
    }
    cols = []
    for c in df.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def build_valid_mask(g: pd.DataFrame) -> np.ndarray:
    valid = np.ones(len(g), dtype=bool)

    if "mean_conf" in g.columns:
        valid &= (g["mean_conf"].astype(float).fillna(-np.inf).to_numpy() >= CONF_THRESH_MEAN)
    if "valid_joints" in g.columns:
        valid &= (g["valid_joints"].astype(float).fillna(-np.inf).to_numpy() >= MIN_VALID_JOINTS)

    return valid


# =========================
# Main
# =========================
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    if df.empty:
        raise RuntimeError(f"Input CSV is empty: {IN_CSV}")

    if "label" not in df.columns or "file" not in df.columns:
        raise RuntimeError("Input CSV must contain at least columns: 'label' and 'file'")

    # Derive grouping + frame index
    df["video_id"] = df["file"].astype(str).apply(parse_video_id)
    df["frame_idx"] = df["file"].astype(str).apply(parse_frame_idx)

    # If frame_idx missing, fall back to original row order within group
    # (still works, but less reliable)
    # We'll sort by frame_idx then by original index.
    df["_orig_i"] = np.arange(len(df), dtype=np.int64)

    # Optional downsample to FPS_TARGET
    stride = max(1, int(round(FPS_ASSUMED / FPS_TARGET)))

    feat_cols = pick_feature_columns(df)
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found in input CSV (besides meta).")

    rows = []

    group_cols = ["label", "video_id"]
    grouped = df.groupby(group_cols, sort=True)

    for (label, video_id), g in tqdm(grouped, desc="Building OpenPose windows", total=len(grouped)):
        g = g.sort_values(["frame_idx", "_orig_i"], ascending=True).reset_index(drop=True)

        # Downsample (approx): keep every 'stride'-th frame
        if stride > 1:
            g = g.iloc[::stride].reset_index(drop=True)

        T = len(g)
        if T < WIN:
            continue

        valid = build_valid_mask(g)

        # Convert features to numpy arrays once (fast)
        feat_mat = {c: g[c].astype(float).to_numpy() for c in feat_cols}

        for start in range(0, T - WIN + 1, STEP):
            end = start + WIN
            valid_frac = float(np.mean(valid[start:end]))
            if valid_frac < MIN_VALID_FRAC:
                continue

            row = {
                "label": label,
                "video_id": video_id,
                "start_frame": start,
                "end_frame": end,
                "valid_frac": valid_frac,
                "fps_target": FPS_TARGET,
                "fps_assumed": FPS_ASSUMED,
                "stride": stride,
            }

            for name, series in feat_mat.items():
                w = series[start:end].copy()

                # Invalidate frames not passing mask -> NaN (like your MediaPipe version)
                m = valid[start:end]
                w[~m] = np.nan

                if np.all(np.isnan(w)):
                    continue

                a = agg_window(w)
                for stat, val in a.items():
                    row[f"{name}__{stat}"] = val

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("Wrote:", OUT_CSV)
    print("Shape:", out.shape)
    print("Classes:", out["label"].nunique() if "label" in out.columns and not out.empty else 0)


if __name__ == "__main__":
    main()
