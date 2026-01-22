#!/usr/bin/env python3
"""
extract_pose_openpose.py

Build a feature CSV from OpenPose JSON outputs with run logging + metrics.

Expected input layout (same as your partner):
OPENPOSE_ROOT/
  squat/
    xxx_keypoints.json
    ...
  push_up/
    ...
  ...

Output:
- OUTPUT_CSV: feature dataset (one row per JSON/frame)
- METRICS_DIR/pose_run_<RUN_ID>.jsonl: append-only run log
- METRICS_DIR/pose_summary_<RUN_ID>.csv: per-class + global summary
- METRICS_DIR/pose_meta_<RUN_ID>.csv: per-file meta (ok/skip/error, timings, conf, etc.)
"""

import os
import json
import time
import socket
import platform
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None


# =========================
# CONFIG (edit here)
# =========================
OPENPOSE_ROOT = Path("/app/openpose/json")  # input root
OUTPUT_CSV = Path("/app/output/openpose_fitness_dataset.csv")

CONF_THRESH_MEAN = 0.5          # skip frame if mean joint confidence < this
MIN_VALID_JOINTS = 6            # require at least N joints with conf>0 (basic sanity)

# If you want "best person" selection in multi-person JSON:
# - "first": use people[0] (fast, matches partner script)
# - "best_mean_conf": choose person with highest mean confidence across used joints
PEOPLE_SELECTION = "best_mean_conf"  # "first" | "best_mean_conf"

# Create output dirs
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("/app/output/metrics/openpose")
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# OpenPose BODY_25 indexing (compatible with your partner mapping)
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


# =========================
# Logging helpers
# =========================
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
RUN_LOG = METRICS_DIR / f"pose_run_{RUN_ID}.jsonl"
RUN_SUMMARY = METRICS_DIR / f"pose_summary_{RUN_ID}.csv"
RUN_META = METRICS_DIR / f"pose_meta_{RUN_ID}.csv"


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proc_mem_mb():
    if psutil is None:
        return None
    try:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_env_info():
    info = {
        "run_id": RUN_ID,
        "ts_utc": now_iso(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "openpose_root": str(OPENPOSE_ROOT),
        "output_csv": str(OUTPUT_CSV),
        "conf_thresh_mean": CONF_THRESH_MEAN,
        "min_valid_joints": MIN_VALID_JOINTS,
        "people_selection": PEOPLE_SELECTION,
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
    }
    # git commit best-effort
    try:
        import subprocess
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


# =========================
# Core feature code
# =========================
def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in degrees for 2D points a,b,c.
    Returns NaN if degenerate.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
        return float("nan")

    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")

    cos_angle = np.dot(ba, bc) / (nba * nbc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def normalize_skeleton(joints_xy: dict) -> dict:
    """
    Translation: subtract hip center.
    Scale: divide by torso length (= norm of shoulder center after translation).
    """
    l_hip = joints_xy.get("l_hip")
    r_hip = joints_xy.get("r_hip")
    l_sh = joints_xy.get("l_shoulder")
    r_sh = joints_xy.get("r_shoulder")

    if l_hip is None or r_hip is None or l_sh is None or r_sh is None:
        # If missing critical joints, return as-is (but likely filtered later)
        return joints_xy

    hip_center = (l_hip + r_hip) / 2.0
    joints_xy = {k: (v - hip_center) for k, v in joints_xy.items()}

    shoulder_center = (joints_xy["l_shoulder"] + joints_xy["r_shoulder"]) / 2.0
    torso = float(np.linalg.norm(shoulder_center))
    torso = max(torso, 1e-6)

    joints_xy = {k: (v / torso) for k, v in joints_xy.items()}
    return joints_xy


def _extract_person_keypoints(person: dict):
    """
    Returns (joints_xy, confs) for the OP_JOINTS subset from one person dict.
    """
    kp = person.get("pose_keypoints_2d", None)
    if kp is None:
        return None, None

    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim != 1 or kp.size < (3 * (max(OP_JOINTS.values()) + 1)):
        return None, None

    joints_xy = {}
    confs = []
    valid = 0
    for name, idx in OP_JOINTS.items():
        x = float(kp[3 * idx])
        y = float(kp[3 * idx + 1])
        c = float(kp[3 * idx + 2])
        joints_xy[name] = np.array([x, y], dtype=np.float32)
        confs.append(c)
        if c > 0:
            valid += 1

    return joints_xy, np.asarray(confs, dtype=np.float32)


def _select_person(data: dict):
    people = data.get("people", [])
    if not people:
        return None

    if PEOPLE_SELECTION == "first":
        return people[0]

    # best_mean_conf across OP_JOINTS subset
    best = None
    best_score = -1.0
    for p in people:
        joints_xy, confs = _extract_person_keypoints(p)
        if confs is None:
            continue
        score = float(np.nanmean(confs)) if confs.size else -1.0
        if score > best_score:
            best_score = score
            best = p
    return best


def process_openpose_json(json_path: Path):
    """
    Return (features_dict, meta_dict) or (None, meta_dict) if skipped.
    """
    t0 = time.perf_counter()
    mem0 = proc_mem_mb()

    meta = {
        "json_path": str(json_path),
        "ts_utc": now_iso(),
        "status": None,
        "reason": None,
        "wall_s": None,
        "mem_mb_start": mem0,
        "mem_mb_end": None,
        "mem_mb_delta": None,
        "mean_conf": None,
        "valid_joints": None,
        "people_count": None,
        "selected_person": PEOPLE_SELECTION,
    }

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [])
        meta["people_count"] = len(people)

        if not people:
            meta["status"] = "skip"
            meta["reason"] = "no_people"
            return None, meta

        person = _select_person(data)
        if person is None:
            meta["status"] = "skip"
            meta["reason"] = "no_valid_person"
            return None, meta

        joints, confs = _extract_person_keypoints(person)
        if joints is None or confs is None:
            meta["status"] = "skip"
            meta["reason"] = "invalid_keypoints_shape"
            return None, meta

        mean_conf = float(np.nanmean(confs)) if confs.size else 0.0
        valid_joints = int(np.sum(confs > 0))

        meta["mean_conf"] = mean_conf
        meta["valid_joints"] = valid_joints

        if valid_joints < MIN_VALID_JOINTS:
            meta["status"] = "skip"
            meta["reason"] = f"too_few_valid_joints<{MIN_VALID_JOINTS}"
            return None, meta

        if mean_conf < CONF_THRESH_MEAN:
            meta["status"] = "skip"
            meta["reason"] = f"mean_conf<{CONF_THRESH_MEAN}"
            return None, meta

        # Normalize
        joints_n = normalize_skeleton(joints)

        # Features
        feats = {}

        # Angles (same set as your MediaPipe engineered features)
        feats["knee_angle_l"] = compute_angle(joints_n["l_hip"], joints_n["l_knee"], joints_n["l_ankle"])
        feats["knee_angle_r"] = compute_angle(joints_n["r_hip"], joints_n["r_knee"], joints_n["r_ankle"])
        feats["hip_angle_l"] = compute_angle(joints_n["l_shoulder"], joints_n["l_hip"], joints_n["l_knee"])
        feats["hip_angle_r"] = compute_angle(joints_n["r_shoulder"], joints_n["r_hip"], joints_n["r_knee"])
        feats["elbow_angle_l"] = compute_angle(joints_n["l_shoulder"], joints_n["l_elbow"], joints_n["l_wrist"])
        feats["elbow_angle_r"] = compute_angle(joints_n["r_shoulder"], joints_n["r_elbow"], joints_n["r_wrist"])

        # Flatten joints + confidences (store both; conf can be useful later)
        for k, v in joints_n.items():
            feats[f"{k}_x"] = float(v[0])
            feats[f"{k}_y"] = float(v[1])
        for (k, _), c in zip(OP_JOINTS.items(), confs.tolist()):
            feats[f"{k}_c"] = float(c)

        feats["mean_conf"] = mean_conf
        feats["valid_joints"] = valid_joints

        meta["status"] = "ok"
        return feats, meta

    except Exception as e:
        meta["status"] = "error"
        meta["reason"] = str(e)
        meta["traceback"] = traceback.format_exc(limit=6)
        return None, meta

    finally:
        wall = time.perf_counter() - t0
        mem1 = proc_mem_mb()
        meta["wall_s"] = float(wall)
        meta["mem_mb_end"] = mem1
        meta["mem_mb_delta"] = (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None


# =========================
# Main
# =========================
def main():
    append_jsonl(RUN_LOG, {"type": "run_start", **get_env_info()})

    rows = []
    meta_rows = []

    if not OPENPOSE_ROOT.exists():
        raise FileNotFoundError(f"OPENPOSE_ROOT not found: {OPENPOSE_ROOT}")

    # Iterate labels/classes (subfolders)
    class_dirs = [p for p in OPENPOSE_ROOT.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)

    total_json = 0
    for class_dir in class_dirs:
        json_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() == ".json"])
        total_json += len(json_files)

    for class_dir in class_dirs:
        label = class_dir.name
        json_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() == ".json"])

        for jp in tqdm(json_files, desc=label):
            feats, meta = process_openpose_json(jp)

            # add label + file for both dataset and meta
            meta["label"] = label
            meta["file"] = jp.name
            meta_rows.append(meta)

            if feats is None:
                append_jsonl(RUN_LOG, {"type": "frame_skip_or_error", **meta})
                continue

            feats["label"] = label
            feats["file"] = jp.name
            feats["json_path"] = str(jp)
            rows.append(feats)

            append_jsonl(RUN_LOG, {"type": "frame_ok", "ts_utc": meta["ts_utc"], "label": label, "file": jp.name,
                                  "mean_conf": meta.get("mean_conf"), "wall_s": meta.get("wall_s")})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(RUN_META, index=False)

    # Summary: counts + skip reasons by class
    if not meta_df.empty:
        ok_mask = meta_df["status"] == "ok"
        summary = meta_df.groupby(["label", "status"]).size().unstack(fill_value=0).reset_index()

        # top skip reasons
        skips = meta_df[meta_df["status"] == "skip"]
        if not skips.empty:
            top_reasons = (
                skips.groupby(["label", "reason"]).size()
                .reset_index(name="count")
                .sort_values(["label", "count"], ascending=[True, False])
            )
        else:
            top_reasons = pd.DataFrame(columns=["label", "reason", "count"])

        # global stats
        global_stats = {
            "label": "__GLOBAL__",
            "ok": int(ok_mask.sum()),
            "skip": int((meta_df["status"] == "skip").sum()),
            "error": int((meta_df["status"] == "error").sum()),
        }
        summary = pd.concat([summary, pd.DataFrame([global_stats])], ignore_index=True, sort=False)
    else:
        summary = pd.DataFrame([{"label": "__GLOBAL__", "ok": 0, "skip": 0, "error": 0}])
        top_reasons = pd.DataFrame(columns=["label", "reason", "count"])

    # write summary (two sections in one CSV is messy; write a clean combined table instead)
    # -> create one CSV with summary rows, and if skip reasons exist append them after a blank line via manual write.
    with open(RUN_SUMMARY, "w", encoding="utf-8") as f:
        f.write("# status_counts_by_label\n")
        summary.to_csv(f, index=False)
        f.write("\n# top_skip_reasons_by_label\n")
        top_reasons.to_csv(f, index=False)

    append_jsonl(RUN_LOG, {"type": "run_end", "ts_utc": now_iso(),
                          "output_csv": str(OUTPUT_CSV),
                          "meta_csv": str(RUN_META),
                          "summary_csv": str(RUN_SUMMARY),
                          "dataset_shape": list(df.shape)})

    print("OpenPose dataset shape:", df.shape)
    print("Saved dataset to:", OUTPUT_CSV)
    print("Saved meta to:", RUN_META)
    print("Saved summary to:", RUN_SUMMARY)
    print("Run log:", RUN_LOG)


if __name__ == "__main__":
    main()
