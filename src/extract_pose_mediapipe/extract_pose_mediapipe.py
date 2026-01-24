import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import json, time, platform, socket, traceback
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    psutil = None

# Config
MANIFEST = "src/extract_pose_mediapipe/manifest.csv"
OUT_DIR = Path("src/output/mediapipe_pose_npz")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FPS_TARGET = 15  # runter-samplen

# Pfad zu deinem .task Modell
MODEL_PATH = "src/extract_pose_mediapipe/pose_landmarker_heavy.task"

# Logging setup
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
METRICS_DIR = Path("src/output/metrics/mediapipe")

# create metrics dir
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG = METRICS_DIR / f"pose_run_{RUN_ID}.jsonl"   # menschenlesbar, append-friendly
RUN_SUMMARY = METRICS_DIR / f"pose_summary_{RUN_ID}.csv"

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def get_env_info():
    info = {
        "run_id": RUN_ID,
        "ts_utc": now_iso(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "fps_target": FPS_TARGET,
        "model_path": MODEL_PATH,
        "mediapipe": getattr(mp, "__version__", "unknown"),
        "opencv": getattr(cv2, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
    }
    # git commit (best effort)
    try:
        import subprocess
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info

def proc_mem_mb():
    if psutil is None:
        return None
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def append_jsonl(path: Path, obj: dict):
    # robust append (one line per record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def _lm_to_arr(lm_list):
    # lm_list: list of 33 landmarks
    # NormalizedLandmark/ Landmark haben x,y,z und oft visibility/presence
    arr = np.zeros((33, 5), dtype=np.float32)  # x,y,z,visibility,presence
    for i, p in enumerate(lm_list):
        arr[i, 0] = getattr(p, "x", np.nan)
        arr[i, 1] = getattr(p, "y", np.nan)
        arr[i, 2] = getattr(p, "z", np.nan)
        arr[i, 3] = getattr(p, "visibility", np.nan)
        arr[i, 4] = getattr(p, "presence", np.nan)
    return arr

def build_video_metrics(
    video_path, label, fps, stride,
    total_frames, ok_frames,
    detect_rate, mean_vis, vis_p50, vis_p90,
    nan_rate_img, wall_s, eff_fps,
    mem0, mem1
):
    return {
        "run_id": RUN_ID,
        "ts_utc": now_iso(),

        "model": "mediapipe_pose",
        "model_variant": "heavy",

        "video_path": video_path,
        "label": label,

        "fps_orig": fps,
        "fps_target": FPS_TARGET,
        "stride": stride,

        "frames_total": total_frames,
        "frames_ok": ok_frames,
        "detect_rate": detect_rate,

        "mean_visibility": mean_vis,
        "vis_p50": vis_p50,
        "vis_p90": vis_p90,
        "nan_rate_xyz": nan_rate_img,

        "wall_s": wall_s,
        "eff_fps": eff_fps,

        "mem_mb_start": mem0,
        "mem_mb_end": mem1,

        "hostname": socket.gethostname(),
        "os": platform.system(),
        "python": platform.python_version(),
    }


def extract_video(video_path: str, landmarker=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / FPS_TARGET)))

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    shared_landmarker = None
    if landmarker is None:
        landmarker = vision.PoseLandmarker.create_from_options(options)
        shared_landmarker = False
    else:
        shared_landmarker = True

    frames_img = []   # (T,33,5) in image coords (normalized)
    frames_world = [] # (T,33,5) in world coords (meters-ish)
    vis = []
    ok_count = 0
    total = 0

    i = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if i % stride != 0:
            i += 1
            continue

        total += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # VIDEO mode braucht monoton steigende timestamps (ms)
        t_ms = int((i / fps) * 1000)

        res = landmarker.detect_for_video(mp_image, t_ms)

        if not res.pose_landmarks:
            frames_img.append(np.full((33, 5), np.nan, dtype=np.float32))
            frames_world.append(np.full((33, 5), np.nan, dtype=np.float32))
            vis.append(0.0)
        else:
            lm_img = res.pose_landmarks[0]         # 33 normalized landmarks
            arr_img = _lm_to_arr(lm_img)

            if getattr(res, "pose_world_landmarks", None) and res.pose_world_landmarks:
                lm_w = res.pose_world_landmarks[0] # 33 world landmarks
                arr_w = _lm_to_arr(lm_w)
            else:
                arr_w = np.full((33, 5), np.nan, dtype=np.float32)

            frames_img.append(arr_img)
            frames_world.append(arr_w)

            mean_vis = float(np.nanmean(arr_img[:, 3])) if np.isfinite(arr_img[:, 3]).any() else 0.0
            vis.append(mean_vis)
            ok_count += 1

        i += 1

    cap.release()
    if not shared_landmarker:
        landmarker.close()

    X_img = np.stack(frames_img, axis=0) if frames_img else np.zeros((0, 33, 5), dtype=np.float32)
    X_world = np.stack(frames_world, axis=0) if frames_world else np.zeros((0, 33, 5), dtype=np.float32)

    vis = np.asarray(vis, dtype=np.float32)

    detect_rate = ok_count / max(1, total)
    mean_vis = float(np.nanmean(vis) if len(vis) else 0.0)

    # robust percentiles (ignore NaN)
    vis_f = vis[np.isfinite(vis)]
    vis_p50 = float(np.percentile(vis_f, 50)) if len(vis_f) else 0.0
    vis_p90 = float(np.percentile(vis_f, 90)) if len(vis_f) else 0.0

    # NaN rate in X_img (after stacking)
    if X_img.size:
        nan_rate_img = float(np.isnan(X_img[:, :, :3]).mean())  # only xyz
    else:
        nan_rate_img = 1.0

    return X_img, X_world, float(fps), stride, float(detect_rate), mean_vis, vis_p50, vis_p90, nan_rate_img, int(total), int(ok_count)


def safe_name(p: str) -> str:
    pp = Path(p)
    cls = pp.parent.name
    stem = pp.stem
    return f"{cls}__{stem}.npz"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)

    # write run header (one-time)
    append_jsonl(RUN_LOG, {"type": "run_start", **get_env_info()})

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    meta_rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        video_path, label = row.video_path, row.label
        out_path = OUT_DIR / safe_name(video_path)

        if out_path.exists():
            # optional: log skip
            append_jsonl(RUN_LOG, {
                "type": "video_skip",
                "ts_utc": now_iso(),
                "video_path": video_path,
                "label": label,
                "npz_path": str(out_path),
                "reason": "exists",
            })
            continue

        t0 = time.perf_counter()
        mem0 = proc_mem_mb()

        try:
            (X_img, X_world, fps, stride, dr, mean_vis,
             vis_p50, vis_p90, nan_rate_img, total_frames, ok_frames) = extract_video(video_path, landmarker=None)

            wall_s = time.perf_counter() - t0
            mem1 = proc_mem_mb()

            np.savez_compressed(
                out_path,
                X_img=X_img,
                X_world=X_world,
                label=label,
                video_path=video_path,
                fps=fps,
                stride=stride,
                detect_rate=dr,
                mean_visibility=mean_vis,
                model_path=MODEL_PATH,
            )

            rec = {
                "type": "video_ok",
                "ts_utc": now_iso(),
                "video_path": video_path,
                "label": label,
                "npz_path": str(out_path),
                "fps_orig": fps,
                "stride": int(stride),
                "fps_sampled": float(fps / stride) if stride else None,
                "frames_processed": int(total_frames),
                "frames_ok": int(ok_frames),
                "detect_rate": float(dr),
                "mean_visibility": float(mean_vis),
                "vis_p50": float(vis_p50),
                "vis_p90": float(vis_p90),
                "nan_rate_xyz": float(nan_rate_img),
                "wall_s": float(wall_s),
                "eff_fps": float(total_frames / wall_s) if wall_s > 0 else None,
                "mem_mb_start": mem0,
                "mem_mb_end": mem1,
                "mem_mb_delta": (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None,
            }

            # minimal additions
            rec["run_id"] = RUN_ID
            rec["model"] = "mediapipe_pose"
            rec["model_variant"] = "heavy"

            append_jsonl(RUN_LOG, rec)
            try:
                requests.post("http://127.0.0.1:8000/metrics", json=rec, timeout=2)
            except Exception:
                pass  # keep extraction running even if backend is down

            meta_rows.append({
                "npz_path": str(out_path),
                "label": label,
                "video_path": video_path,
                "frames_T": int(X_img.shape[0]),
                "fps_orig": fps,
                "stride": stride,
                "detect_rate": dr,
                "mean_visibility": mean_vis,
                "vis_p50": vis_p50,
                "vis_p90": vis_p90,
                "nan_rate_xyz": nan_rate_img,
                "wall_s": wall_s,
                "eff_fps": rec["eff_fps"],
                "mem_mb_delta": rec["mem_mb_delta"],
            })

        except Exception as e:
            wall_s = time.perf_counter() - t0
            mem1 = proc_mem_mb()
            tb = traceback.format_exc(limit=5)

            rec = {
                "type": "video_error",
                "ts_utc": now_iso(),
                "video_path": video_path,
                "label": label,
                "error": str(e),
                "traceback": tb,
                "wall_s": float(wall_s),
                "mem_mb_start": mem0,
                "mem_mb_end": mem1,
            }
            append_jsonl(RUN_LOG, rec)

            meta_rows.append({
                "npz_path": "",
                "label": label,
                "video_path": video_path,
                "error": str(e),
            })

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(METRICS_DIR / "pose_meta.csv", index=False)

    # additionally store run-scoped summary with timestamp in filename
    meta.to_csv(RUN_SUMMARY, index=False)
    append_jsonl(RUN_LOG, {"type": "run_end", "ts_utc": now_iso(), "summary_csv": str(RUN_SUMMARY)})

    print(f"Wrote pose_meta.csv and {RUN_LOG} and {RUN_SUMMARY}")

if __name__ == "__main__":
    main()