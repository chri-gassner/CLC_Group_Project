"""
mediapipe_pose_npz_batch.py

Batch extraction of MediaPipe Pose landmarks from videos listed in a manifest CSV.

This version is deliberately implemented in the "safe" way:
- A new PoseLandmarker is created per video and closed per video.
  This avoids the VIDEO-mode monotonic-timestamp issue across multiple videos.

What it does:
- Reads MANIFEST (CSV) with at least columns: `video_path`, `label`
- Samples frames by stride to approximate FPS_TARGET
- Extracts:
  - normalized image landmarks (33x5: x,y,z,visibility,presence)
  - world landmarks (33x5)
- Writes per-video NPZ files into OUT_DIR
- Writes JSONL run log + CSV summary into METRICS_DIR
- Optionally posts per-video metrics to a Cloud Run ingestion endpoint using an ID token

Notes:
- VIDEO running mode requires monotonically increasing timestamps *within a single landmarker instance*.
  Because we create a new landmarker per video, timestamps can safely restart at 0 for each video.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import psutil
except ImportError:
    psutil = None


# ----------------------------
# Config
# ----------------------------

DEBUG = True

MANIFEST = "manifest.csv"

OUT_DIR = Path("../output/mediapipe_pose_npz_fps_15_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS_TARGET = 15  # Approximate downsampling target FPS using stride.
MODEL_PATH = "pose_landmarker_heavy.task"

RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
METRICS_DIR = Path("../output/metrics/mediapipe")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

INGEST_URL = os.environ.get("INGEST_URL", None)

RUN_LOG = METRICS_DIR / f"pose_run_{RUN_ID}.jsonl"
RUN_SUMMARY = METRICS_DIR / f"pose_summary_{RUN_ID}.csv"

SOURCE_NAME = "client_mediapipe"

if not INGEST_URL:
    print("Warning: INGEST_URL not set; API posting disabled.")
    print("Set INGEST_URL environment variable to enable posting.")


#print current working directory
print("Current working directory:", os.getcwd())

# ----------------------------
# Auth + HTTP
# ----------------------------

def get_bearer_token(audience_url: str) -> str:
    """
    Create an ID token for a Cloud Run service.

    Parameters
    ----------
    audience_url:
        Cloud Run base URL (target audience), e.g. "https://<service>...run.app"

    Returns
    -------
    str
        Bearer token for Authorization header.
    """
    creds = service_account.IDTokenCredentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        target_audience=audience_url,
    )
    creds.refresh(Request())
    return creds.token


def send_request_api(url: str, json_data: Dict[str, Any], token: str) -> requests.Response:
    """
    Send a POST request to the ingestion API.

    In DEBUG mode, prints response status and body.

    Notes
    -----
    - Uses a short timeout to avoid blocking long batch runs if the API is slow.
    """
    resp = requests.post(
        url,
        json=json_data,
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    if DEBUG:
        print(f"Response status code: {resp.status_code}")
        print(f"Response text: {resp.text}")
    return resp


AUDIENCE = INGEST_URL.split("/metrics")[0]
TOKEN = get_bearer_token(AUDIENCE)


# ----------------------------
# Utilities
# ----------------------------

def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format (seconds resolution)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_env_info() -> Dict[str, Any]:
    """
    Collect reproducibility metadata about the runtime environment.

    Returns
    -------
    dict
        Includes library versions and (best effort) git commit hash.
    """
    info: Dict[str, Any] = {
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

    try:
        import subprocess
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        info["git_commit"] = None

    return info


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append a single JSON record to a JSONL file (one object per line).
    fsync is used to reduce data loss on crashes.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def proc_mem_mb() -> Optional[float]:
    """Return process resident memory (RSS) in MB (best effort)."""
    if psutil is None:
        return None
    p = psutil.Process(os.getpid())
    return float(p.memory_info().rss / (1024 * 1024))


def cpu_time_s() -> Optional[float]:
    """Return process CPU time (user + system) in seconds (best effort)."""
    if psutil is None:
        return None
    t = psutil.Process(os.getpid()).cpu_times()
    return float(t.user + t.system)


def sample_proc_stats() -> Dict[str, Any]:
    """
    Lightweight process stats snapshot (best effort).
    Used for approximating peak RSS during extraction.
    """
    if psutil is None:
        return {}
    p = psutil.Process(os.getpid())
    return {
        "rss_mb": float(p.memory_info().rss / (1024 * 1024)),
        "num_threads": int(p.num_threads()),
    }


def summarize_latencies_ms(lat_ms_list: Sequence[float]) -> Dict[str, Any]:
    """
    Compute latency percentiles for a sequence of per-frame latencies (milliseconds).

    Returns
    -------
    dict with keys:
        median_ms, p90_ms, p99_ms, n
    """
    if not lat_ms_list:
        return {"median_ms": None, "p90_ms": None, "p99_ms": None, "n": 0}
    a = np.asarray(lat_ms_list, dtype=np.float64)
    return {
        "median_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "n": int(a.size),
    }


# ----------------------------
# MediaPipe helpers
# ----------------------------

def landmarks_to_array(lm_list: Sequence[Any]) -> np.ndarray:
    """
    Convert a list of MediaPipe Pose landmarks into a (33, 5) float32 array.

    Columns: x, y, z, visibility, presence

    Missing fields are filled with NaN.
    """
    arr = np.zeros((33, 5), dtype=np.float32)
    for i, p in enumerate(lm_list):
        arr[i, 0] = getattr(p, "x", np.nan)
        arr[i, 1] = getattr(p, "y", np.nan)
        arr[i, 2] = getattr(p, "z", np.nan)
        arr[i, 3] = getattr(p, "visibility", np.nan)
        arr[i, 4] = getattr(p, "presence", np.nan)
    return arr


def create_landmarker() -> vision.PoseLandmarker:
    """
    Create a PoseLandmarker instance configured for VIDEO mode.

    Important
    ---------
    In VIDEO mode, timestamps must be monotonically increasing within a given
    landmarker instance. This script creates a new instance per video to keep
    timestamp logic simple and correct.
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_video(video_path: str) -> Tuple[
    np.ndarray, np.ndarray, float, int,
    float, float, float, float, float,
    int, int, List[float], Optional[float]
]:
    """
    Extract landmarks from one video.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    tuple:
        X_img (T,33,5),
        X_world (T,33,5),
        fps_orig,
        stride,
        detect_rate,
        mean_visibility,
        vis_p50,
        vis_p90,
        nan_rate_xyz,
        frames_processed (sampled frames),
        frames_ok,
        infer_lat_ms (list),
        rss_peak_mb
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(fps / FPS_TARGET)))

    landmarker = create_landmarker()  # per-video to avoid monotonic timestamp issues
    try:
        frames_img: List[np.ndarray] = []
        frames_world: List[np.ndarray] = []
        vis: List[float] = []

        ok_count = 0
        total = 0

        infer_lat_ms: List[float] = []
        rss_peak_mb: Optional[float] = None

        i = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Approximate sampling using stride
            if i % stride != 0:
                i += 1
                continue

            total += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # VIDEO mode timestamp in milliseconds; monotonic within this video/landmarker
            t_ms = int((i / fps) * 1000)

            t_inf0 = time.perf_counter()
            res = landmarker.detect_for_video(mp_image, t_ms)
            t_inf1 = time.perf_counter()
            infer_lat_ms.append((t_inf1 - t_inf0) * 1000.0)

            st = sample_proc_stats()
            if st.get("rss_mb") is not None:
                rss_peak_mb = st["rss_mb"] if rss_peak_mb is None else max(rss_peak_mb, st["rss_mb"])

            if not res.pose_landmarks:
                frames_img.append(np.full((33, 5), np.nan, dtype=np.float32))
                frames_world.append(np.full((33, 5), np.nan, dtype=np.float32))
                vis.append(0.0)
            else:
                arr_img = landmarks_to_array(res.pose_landmarks[0])

                if getattr(res, "pose_world_landmarks", None) and res.pose_world_landmarks:
                    arr_w = landmarks_to_array(res.pose_world_landmarks[0])
                else:
                    arr_w = np.full((33, 5), np.nan, dtype=np.float32)

                frames_img.append(arr_img)
                frames_world.append(arr_w)

                mean_vis = float(np.nanmean(arr_img[:, 3])) if np.isfinite(arr_img[:, 3]).any() else 0.0
                vis.append(mean_vis)
                ok_count += 1

            i += 1

    finally:
        cap.release()
        landmarker.close()
    landmarker.close()
    X_img = np.stack(frames_img, axis=0) if frames_img else np.zeros((0, 33, 5), dtype=np.float32)
    X_world = np.stack(frames_world, axis=0) if frames_world else np.zeros((0, 33, 5), dtype=np.float32)

    vis_arr = np.asarray(vis, dtype=np.float32)

    detect_rate = float(ok_count / max(1, total))
    mean_vis = float(np.nanmean(vis_arr) if len(vis_arr) else 0.0)

    vis_f = vis_arr[np.isfinite(vis_arr)]
    vis_p50 = float(np.percentile(vis_f, 50)) if len(vis_f) else 0.0
    vis_p90 = float(np.percentile(vis_f, 90)) if len(vis_f) else 0.0

    nan_rate_xyz = float(np.isnan(X_img[:, :, :3]).mean()) if X_img.size else 1.0
    rss_peak_mb = float(rss_peak_mb) if rss_peak_mb is not None else None

    return (
        X_img, X_world, fps, stride,
        detect_rate, mean_vis, vis_p50, vis_p90, nan_rate_xyz,
        int(total), int(ok_count),
        infer_lat_ms, rss_peak_mb,
    )


def safe_name(video_path: str) -> str:
    """
    Build a deterministic NPZ file name:
    "<class_dir>__<video_stem>.npz"
    """
    p = Path(video_path)
    return f"{p.parent.name}__{p.stem}.npz"


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    """
    Run batch extraction for all videos listed in the manifest CSV.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)

    append_jsonl(RUN_LOG, {"type": "run_start", **get_env_info()})

    meta_rows: List[Dict[str, Any]] = []

    # tqdm prints to stdout explicitly (useful in some container logging setups)
    for row in tqdm(df.itertuples(index=False), total=len(df), file=sys.stdout):
        video_path, label = row.video_path, row.label
        out_path = OUT_DIR / safe_name(video_path)

        if out_path.exists():
            if DEBUG:
                print(f"Skipping existing output: {out_path}")
            log = {
                    "type": "video_skip",
                    "ts_utc": now_iso(),
                    "video_path": video_path,
                    "label": label,
                    "npz_path": str(out_path),
                    "reason": "exists",
                }

            append_jsonl(
                RUN_LOG,
                log
            )
            # send to api as well
            if INGEST_URL:
                try:
                    send_request_api(INGEST_URL, log, TOKEN)
                except Exception as e:
                    append_jsonl(
                        RUN_LOG,
                        {
                            "type": "api_error",
                            "ts_utc": now_iso(),
                            "video_path": video_path,
                            "label": label,
                            "error": str(e),
                        },
                    )
            continue

        t0 = time.perf_counter()
        mem0 = proc_mem_mb()
        cpu0 = cpu_time_s()

        try:
            (
                X_img, X_world, fps, stride,
                dr, mean_vis, vis_p50, vis_p90, nan_rate_img,
                total_frames, ok_frames,
                infer_lat_ms, rss_peak_mb,
            ) = extract_video(video_path)

            wall_s = float(time.perf_counter() - t0)
            mem1 = proc_mem_mb()
            cpu1 = cpu_time_s()

            cpu_s = (cpu1 - cpu0) if (cpu0 is not None and cpu1 is not None) else None
            cpu_util_pct = (100.0 * cpu_s / wall_s) if (cpu_s is not None and wall_s > 0) else None

            lat_stats = summarize_latencies_ms(infer_lat_ms)

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

            rec: Dict[str, Any] = {
                "type": "video_ok",
                "source": SOURCE_NAME,
                "ts_utc": now_iso(),
                "video_path": video_path,
                "label": label,
                "npz_path": str(out_path),
                "fps_orig": float(fps),
                "stride": int(stride),
                "fps_sampled": float(fps / stride) if stride else None,
                "frames_processed": int(total_frames),
                "frames_ok": int(ok_frames),
                "detect_rate": float(dr),
                "mean_visibility": float(mean_vis),
                "vis_p50": float(vis_p50),
                "vis_p90": float(vis_p90),
                "nan_rate_xyz": float(nan_rate_img),
                "wall_s": wall_s,
                "eff_fps": float(total_frames / wall_s) if wall_s > 0 else None,
                "mem_mb_start": mem0,
                "mem_mb_end": mem1,
                "mem_mb_delta": (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None,
                "rss_peak_mb": rss_peak_mb,
                "cpu_time_s": cpu_s,
                "cpu_util_pct": cpu_util_pct,
                "infer_lat_median_ms": lat_stats["median_ms"],
                "infer_lat_p90_ms": lat_stats["p90_ms"],
                "infer_lat_p99_ms": lat_stats["p99_ms"],
                "infer_lat_n": lat_stats["n"],
            }

            append_jsonl(RUN_LOG, rec)

            if DEBUG:
                print(f"Processed video OK: {video_path}")

            # Keep API posting best-effort; do not fail the whole run if the API fails.
            if INGEST_URL:
                try:
                    send_request_api(INGEST_URL, rec, TOKEN)
                except Exception as e:
                    append_jsonl(
                        RUN_LOG,
                        {
                            "type": "api_error",
                            "ts_utc": now_iso(),
                            "video_path": video_path,
                            "label": label,
                            "error": str(e),
                        },
                    )

            meta_rows.append(
                {
                    "npz_path": str(out_path),
                    "label": label,
                    "video_path": video_path,
                    "frames_T": int(X_img.shape[0]),
                    "fps_orig": float(fps),
                    "stride": int(stride),
                    "detect_rate": float(dr),
                    "mean_visibility": float(mean_vis),
                    "vis_p50": float(vis_p50),
                    "vis_p90": float(vis_p90),
                    "nan_rate_xyz": float(nan_rate_img),
                    "wall_s": wall_s,
                    "eff_fps": rec["eff_fps"],
                    "mem_mb_delta": rec["mem_mb_delta"],
                    "rss_peak_mb": rss_peak_mb,
                    "cpu_util_pct": cpu_util_pct,
                    "infer_lat_median_ms": lat_stats["median_ms"],
                    "infer_lat_p90_ms": lat_stats["p90_ms"],
                    "infer_lat_p99_ms": lat_stats["p99_ms"],
                }
            )

        except Exception as e:
            wall_s = float(time.perf_counter() - t0)
            mem1 = proc_mem_mb()
            tb = traceback.format_exc(limit=5)

            rec = {
                "type": "video_error",
                "source": SOURCE_NAME,
                "ts_utc": now_iso(),
                "video_path": video_path,
                "label": label,
                "error": str(e),
                "traceback": tb,
                "wall_s": wall_s,
                "mem_mb_start": mem0,
                "mem_mb_end": mem1,
            }
            append_jsonl(RUN_LOG, rec)

            if DEBUG:
                print(f"Error processing video: {video_path}")
                # print(tb)

            # send to api as well
            if INGEST_URL:
                try:
                    send_request_api(INGEST_URL, rec, TOKEN)
                except Exception as e:
                    append_jsonl(
                        RUN_LOG,
                        {
                            "type": "api_error",
                            "ts_utc": now_iso(),
                            "video_path": video_path,
                            "label": label,
                            "error": str(e),
                        },
                    )


            meta_rows.append(
                {
                    "npz_path": "",
                    "label": label,
                    "video_path": video_path,
                    "error": str(e),
                }
            )

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(METRICS_DIR / "pose_meta.csv", index=False)
    meta.to_csv(RUN_SUMMARY, index=False)

    append_jsonl(RUN_LOG, {"type": "run_end", "ts_utc": now_iso(), "summary_csv": str(RUN_SUMMARY)})

    # Console summary (best effort)
    ok = meta[meta.get("error").isna()] if "error" in meta.columns else meta

    def smean(col: str) -> Optional[float]:
        return float(ok[col].mean()) if col in ok.columns and len(ok) else None

    def smedian(col: str) -> Optional[float]:
        return float(ok[col].median()) if col in ok.columns and len(ok) else None

    summary = {
        "videos_total": int(len(meta)),
        "videos_ok": int(len(ok)),
        "detect_rate_mean": smean("detect_rate"),
        "wall_s_median": smedian("wall_s"),
        "eff_fps_mean": smean("eff_fps"),
        "infer_lat_median_ms_median": smedian("infer_lat_median_ms"),
        "infer_lat_p90_ms_median": smedian("infer_lat_p90_ms"),
        "infer_lat_p99_ms_median": smedian("infer_lat_p99_ms"),
        "cpu_util_pct_mean": smean("cpu_util_pct"),
        "rss_peak_mb_max": float(ok["rss_peak_mb"].max()) if "rss_peak_mb" in ok.columns and len(ok) else None,
    }

    print("=== RUN SUMMARY (accuracy proxies + latency + resource usage) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"Wrote pose_meta.csv and {RUN_LOG} and {RUN_SUMMARY}")


if __name__ == "__main__":
    main()
