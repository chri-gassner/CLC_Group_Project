#!/usr/bin/env python3
"""
extract_pose_openpose.py

OpenPose extraction/benchmark aligned to your MediaPipe schema + auth + API posting.

Key design:
- Uses MANIFEST CSV with at least: `video_path`, `label` (same as MediaPipe).
- Runs OpenPose per video, writes per-frame JSON into OUT_DIR/<class_dir>__<video_stem>/
- Logs:
  - JSONL run log (RUN_LOG)
  - CSV run summary (RUN_SUMMARY)
  - CSV meta (pose_meta.csv) like your MediaPipe script
- Posts per-video records (ok/skip/error) best-effort to INGEST_URL with Cloud Run ID token auth.

Field alignment notes:
- We can match most Meta/Run fields 1:1.
- OpenPose has no "world landmarks" and no per-frame infer latency -> infer_lat_* are None.
- "mean_visibility"/"vis_p50"/"vis_p90" are mapped to OpenPose keypoint confidence stats.
- "nan_rate_xyz" is approximated from missing/zeroed keypoints (OpenPose does not emit NaN).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore

# Auth (Cloud Run ID token)
try:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
except Exception:
    Request = None  # type: ignore
    service_account = None  # type: ignore


# ----------------------------
# Config (match MediaPipe style)
# ----------------------------

DEBUG = True

MANIFEST = os.environ.get("MANIFEST", "manifest.csv")

# Output directory for per-video JSON frame dumps
OUT_DIR = Path(os.environ.get("OUT_DIR", "../output/openpose_json"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics directory for JSONL + CSV summary (aligned)
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
METRICS_DIR = Path(os.environ.get("METRICS_DIR", "../output/metrics/openpose"))
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG = METRICS_DIR / f"openpose_run_{RUN_ID}.jsonl"
RUN_SUMMARY = METRICS_DIR / f"openpose_summary_{RUN_ID}.csv"

SOURCE_NAME = os.environ.get("SOURCE_NAME", "client_openpose")

# OpenPose
OPENPOSE_BIN = os.environ.get("OPENPOSE_BIN", "/openpose/build/examples/openpose/openpose.bin")
OPENPOSE_CWD = os.environ.get("OPENPOSE_CWD", "/openpose")  # where models/ are expected
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "body25")   # "body25" or "coco"
NUMBER_PEOPLE_MAX = int(os.environ.get("NUMBER_PEOPLE_MAX", "1"))

# Progress polling
PROGRESS_POLL_S = float(os.environ.get("PROGRESS_POLL_S", "0.5"))

# Optional API posting (same semantics as your MediaPipe)
INGEST_URL = os.environ.get("INGEST_URL", "").strip()

if not INGEST_URL:
    print("Warning: INGEST_URL not set; API posting disabled.")
    print("Set INGEST_URL environment variable to enable posting.")


# ----------------------------
# Auth + HTTP (aligned)
# ----------------------------

def get_bearer_token(audience_url: str) -> str:
    """
    Create an ID token for a Cloud Run service.
    Requires GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON.
    """
    if service_account is None or Request is None:
        raise RuntimeError("google-auth not installed (pip install google-auth).")
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set.")
    creds = service_account.IDTokenCredentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        target_audience=audience_url,
    )
    creds.refresh(Request())
    return creds.token


def send_request_api(url: str, json_data: Dict[str, Any], token: str) -> Any:
    """
    Best-effort POST to ingestion API. In DEBUG mode prints status and body.
    """
    if DEBUG:
        print(f"Sending POST to {url} with json data keys: {list(json_data.keys())}")
    if requests is None:
        raise RuntimeError("requests not installed.")
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


def get_token_if_enabled() -> Optional[str]:
    if not INGEST_URL:
        return None
    # same convention you used: audience = base before "/metrics"
    audience = INGEST_URL.split("/metrics")[0]
    return get_bearer_token(audience)


# ----------------------------
# Utilities (aligned)
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def proc_mem_mb() -> Optional[float]:
    if psutil is None:
        return None
    p = psutil.Process(os.getpid())
    return float(p.memory_info().rss / (1024 * 1024))


def cpu_time_s() -> Optional[float]:
    if psutil is None:
        return None
    t = psutil.Process(os.getpid()).cpu_times()
    return float(t.user + t.system)


def sample_proc_stats() -> Dict[str, Any]:
    if psutil is None:
        return {}
    p = psutil.Process(os.getpid())
    return {
        "rss_mb": float(p.memory_info().rss / (1024 * 1024)),
        "num_threads": int(p.num_threads()),
    }


def summarize_latencies_ms(lat_ms_list: Sequence[float]) -> Dict[str, Any]:
    # OpenPose does not provide per-frame inference timing via CLI reliably.
    # Keep function for schema parity.
    if not lat_ms_list:
        return {"median_ms": None, "p90_ms": None, "p99_ms": None, "n": 0}
    a = np.asarray(lat_ms_list, dtype=np.float64)
    return {
        "median_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "n": int(a.size),
    }


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "run_id": RUN_ID,
        "ts_utc": now_iso(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "model": "openpose",
        "model_variant": MODEL_VARIANT,
        "openpose_bin": OPENPOSE_BIN,
        "out_dir": str(OUT_DIR),
        "manifest": MANIFEST,
    }
    if cv2 is not None:
        info["opencv"] = getattr(cv2, "__version__", "unknown")
    info["numpy"] = getattr(np, "__version__", "unknown")
    info["pandas"] = getattr(pd, "__version__", "unknown")

    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def safe_video_fps(path: str) -> Optional[float]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return float(fps)
    return None


# ----------------------------
# OpenPose JSON parsing helpers
# ----------------------------

def _keypoints_from_person(person: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Return keypoints array as shape (K,3) [x,y,score] for BODY_25 or COCO.
    OpenPose JSON stores flat list under 'pose_keypoints_2d'.
    """
    flat = person.get("pose_keypoints_2d", None)
    if not flat or not isinstance(flat, list):
        return None
    if len(flat) % 3 != 0:
        return None
    k = len(flat) // 3
    arr = np.asarray(flat, dtype=np.float32).reshape(k, 3)
    return arr


def _frame_stats_from_json(j: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Per-frame stats:
    - detected: at least 1 person
    - mean_conf: mean score across keypoints (if detected)
    - missing_rate: fraction of keypoints considered "missing" (heuristic)
      missing if (x==0 and y==0) OR score==0
    """
    people = j.get("people", [])
    if not people:
        return (False, None, None)

    kp = _keypoints_from_person(people[0])
    if kp is None or kp.size == 0:
        return (True, None, None)

    scores = kp[:, 2]
    mean_conf = float(np.mean(scores)) if scores.size else None

    missing = ((kp[:, 0] == 0.0) & (kp[:, 1] == 0.0)) | (scores == 0.0)
    missing_rate = float(np.mean(missing)) if missing.size else None

    return (True, mean_conf, missing_rate)


def safe_name(video_path: str) -> str:
    """
    Deterministic per-video output folder name:
    "<class_dir>__<video_stem>"
    """
    p = Path(video_path)
    return f"{p.parent.name}__{p.stem}"


# ----------------------------
# Core: run OpenPose per video
# ----------------------------

def run_openpose_on_video(video_path: str, label: Any) -> Dict[str, Any]:
    out_dir = OUT_DIR / safe_name(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = now_iso()
    fps_orig = safe_video_fps(video_path)

    base_rec: Dict[str, Any] = {
        "type": None,  # set later
        "source": SOURCE_NAME,
        "ts_utc": ts,
        "run_id": RUN_ID,
        "model": "openpose",
        "model_variant": MODEL_VARIANT,
        "video_path": video_path,
        "label": label,
        "out_dir": str(out_dir),
        "fps_orig": float(fps_orig) if fps_orig is not None else None,
        # OpenPose CLI processes full video; no stride sampling here.
        "stride": 1,
        "fps_sampled": float(fps_orig) if fps_orig is not None else None,
    }

    if not os.path.exists(video_path):
        return {**base_rec, "type": "video_error", "error": f"Video not found: {video_path}"}

    if not os.path.exists(OPENPOSE_BIN):
        return {**base_rec, "type": "video_error", "error": f"OPENPOSE_BIN not found: {OPENPOSE_BIN}"}

    # If already exists and has JSON -> skip (aligned behavior)
    existing = glob.glob(str(out_dir / "*.json"))
    if existing:
        return {
            **base_rec,
            "type": "video_skip",
            "npz_path": None,  # schema parity
            "reason": "exists",
            "frames_processed": int(len(existing)),
        }

    # Build command
    cmd = [
        OPENPOSE_BIN,
        "--video", video_path,
        "--write_json", str(out_dir),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", "160x80",
        "--number_people_max", str(NUMBER_PEOPLE_MAX),
        "--profile_speed","999"
    ]
    if MODEL_VARIANT.lower() == "coco":
        cmd += ["--model_pose", "COCO"]
    else:
        cmd += ["--model_pose", "BODY_25"]

    t0 = time.perf_counter()
    mem0 = proc_mem_mb()
    cpu0 = cpu_time_s()
    rss_peak_mb: Optional[float] = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=OPENPOSE_CWD,
        )
    except Exception as e:
        return {**base_rec, "type": "video_error", "error": f"subprocess.Popen failed: {e}"}

    p_psutil = None
    if psutil is not None:
        try:
            p_psutil = psutil.Process(proc.pid)
        except Exception:
            p_psutil = None

    last_count = 0
    # Unknown total -> use tqdm without total
    with tqdm(
        desc=Path(video_path).name,
        unit="frame",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        while True:
            rc = proc.poll()

            json_count = len(glob.glob(str(out_dir / "*.json")))
            delta = json_count - last_count
            if delta > 0:
                pbar.update(delta)
                last_count = json_count

            st = sample_proc_stats()
            if st.get("rss_mb") is not None:
                rss_peak_mb = float(st["rss_mb"]) if rss_peak_mb is None else max(rss_peak_mb, float(st["rss_mb"]))

            wall_s_now = float(time.perf_counter() - t0)
            eff_fps_now = (json_count / wall_s_now) if wall_s_now > 0 else 0.0
            pbar.set_postfix({
                "json": json_count,
                "eff_fps": f"{eff_fps_now:.1f}",
                "rss_mb": f"{(rss_peak_mb or 0):.0f}",
                "rc": rc if rc is not None else "-",
            })

            if rc is not None:
                break
            time.sleep(PROGRESS_POLL_S)

        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""

    wall_s = float(time.perf_counter() - t0)
    mem1 = proc_mem_mb()
    cpu1 = cpu_time_s()

    cpu_s = (cpu1 - cpu0) if (cpu0 is not None and cpu1 is not None) else None
    cpu_util_pct = (100.0 * cpu_s / wall_s) if (cpu_s is not None and wall_s > 0) else None

    stdout_tail = (out or "")[-2000:]
    stderr_tail = (err or "")[-2000:]

    json_files = sorted(glob.glob(str(out_dir / "*.json")))
    num_frames = int(len(json_files))

    json_files = sorted(glob.glob(str(out_dir / "*.json")))
    num_frames = int(len(json_files))

    # --- INSERT START ---
    actual_model_lat = None
    if err:
        for line in err.splitlines():
            if "ms / frame" in line:
                try:
                    parts = line.split()
                    # Finds the number immediately preceding "ms"
                    actual_model_lat = float(parts[parts.index("ms") - 1])
                    break
                except (ValueError, IndexError):
                    continue
    
    # Use parsed latency if found; fallback to wall-clock average
    lat_val = actual_model_lat if actual_model_lat else (wall_s / max(1, num_frames)) * 1000
    lat_stats = summarize_latencies_ms([lat_val] if num_frames > 0 else [])
    # --- INSERT END ---

    # Parse detection + confidence stats
    detected_frames = 0

    # Parse detection + confidence stats
    detected_frames = 0
    confs: List[float] = []
    miss_rates: List[float] = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                j = json.load(f)
            detected, mean_conf, miss_rate = _frame_stats_from_json(j)
            if detected:
                detected_frames += 1
            if mean_conf is not None:
                confs.append(float(mean_conf))
            if miss_rate is not None:
                miss_rates.append(float(miss_rate))
        except Exception:
            # ignore corrupt frame json; do not crash
            pass

    detect_rate = float(detected_frames / max(1, num_frames)) if num_frames > 0 else None

    conf_arr = np.asarray(confs, dtype=np.float32)
    mean_conf = float(np.mean(conf_arr)) if conf_arr.size else 0.0
    vis_p50 = float(np.percentile(conf_arr, 50)) if conf_arr.size else 0.0
    vis_p90 = float(np.percentile(conf_arr, 90)) if conf_arr.size else 0.0

    # "nan_rate_xyz" proxy: mean missing keypoint rate across frames where computed
    mr_arr = np.asarray(miss_rates, dtype=np.float32)
    nan_rate_xyz = float(np.mean(mr_arr)) if mr_arr.size else (1.0 if num_frames == 0 else 0.0)

    eff_fps = float(num_frames / wall_s) if (num_frames > 0 and wall_s > 0) else None

    # OpenPose has no per-frame infer latency list here
    #lat_stats = summarize_latencies_ms([])

    # Decide ok vs error based on return code and output presence
    if proc.returncode != 0:
        return {
            **base_rec,
            "type": "video_error",
            "returncode": int(proc.returncode),
            "frames_processed": num_frames,
            "frames_ok": int(detected_frames),
            "detect_rate": detect_rate,
            "mean_visibility": float(mean_conf),
            "vis_p50": float(vis_p50),
            "vis_p90": float(vis_p90),
            "nan_rate_xyz": float(nan_rate_xyz),
            "wall_s": wall_s,
            "eff_fps": eff_fps,
            "mem_mb_start": mem0,
            "mem_mb_end": mem1,
            "mem_mb_delta": (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None,
            "rss_peak_mb": float(rss_peak_mb) if rss_peak_mb is not None else None,
            "cpu_time_s": cpu_s,
            "cpu_util_pct": cpu_util_pct,
            "infer_lat_median_ms": lat_stats["median_ms"],
            "infer_lat_p90_ms": lat_stats["p90_ms"],
            "infer_lat_p99_ms": lat_stats["p99_ms"],
            "infer_lat_n": lat_stats["n"],
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "error": "openpose_returncode_nonzero",
        }

    if num_frames == 0:
        return {
            **base_rec,
            "type": "video_error",
            "returncode": int(proc.returncode),
            "frames_processed": 0,
            "frames_ok": 0,
            "detect_rate": None,
            "mean_visibility": 0.0,
            "vis_p50": 0.0,
            "vis_p90": 0.0,
            "nan_rate_xyz": 1.0,
            "wall_s": wall_s,
            "eff_fps": None,
            "mem_mb_start": mem0,
            "mem_mb_end": mem1,
            "mem_mb_delta": (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None,
            "rss_peak_mb": float(rss_peak_mb) if rss_peak_mb is not None else None,
            "cpu_time_s": cpu_s,
            "cpu_util_pct": cpu_util_pct,
            "infer_lat_median_ms": lat_stats["median_ms"],
            "infer_lat_p90_ms": lat_stats["p90_ms"],
            "infer_lat_p99_ms": lat_stats["p99_ms"],
            "infer_lat_n": lat_stats["n"],
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "error": "zero_json_frames",
        }

    # OK
    return {
        **base_rec,
        "type": "video_ok",
        "returncode": int(proc.returncode),
        "frames_processed": num_frames,
        "frames_ok": int(detected_frames),
        "detect_rate": float(detect_rate) if detect_rate is not None else None,
        "mean_visibility": float(mean_conf),
        "vis_p50": float(vis_p50),
        "vis_p90": float(vis_p90),
        "nan_rate_xyz": float(nan_rate_xyz),
        "wall_s": wall_s,
        "eff_fps": eff_fps,
        "mem_mb_start": mem0,
        "mem_mb_end": mem1,
        "mem_mb_delta": (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None,
        "rss_peak_mb": float(rss_peak_mb) if rss_peak_mb is not None else None,
        "cpu_time_s": cpu_s,
        "cpu_util_pct": cpu_util_pct,
        "infer_lat_median_ms": lat_stats["median_ms"],
        "infer_lat_p90_ms": lat_stats["p90_ms"],
        "infer_lat_p99_ms": lat_stats["p99_ms"],
        "infer_lat_n": lat_stats["n"],
    }


# ----------------------------
# Main (aligned to MediaPipe)
# ----------------------------

def main() -> None:
    df = pd.read_csv(MANIFEST)
    if "video_path" not in df.columns or "label" not in df.columns:
        raise ValueError("MANIFEST must contain columns: video_path, label")

    token: Optional[str] = None
    if INGEST_URL:
        token = get_token_if_enabled()

    append_jsonl(RUN_LOG, {"type": "run_start", **get_env_info()})

    meta_rows: List[Dict[str, Any]] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), file=sys.stdout):
        video_path = str(getattr(row, "video_path"))
        label = getattr(row, "label")

        try:
            rec = run_openpose_on_video(video_path, label)
            append_jsonl(RUN_LOG, rec)

            # best-effort API posting (ok/skip/error)
            if INGEST_URL and token is not None:
                try:
                    send_request_api(INGEST_URL, rec, token)
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

            # meta row (keep very similar to MP meta_rows)
            meta_rows.append(
                {
                    "out_dir": rec.get("out_dir", ""),
                    "label": label,
                    "video_path": video_path,
                    "error": rec.get("error"),
                    "fps_orig": rec.get("fps_orig"),
                    "stride": rec.get("stride"),
                    "detect_rate": rec.get("detect_rate"),
                    "mean_visibility": rec.get("mean_visibility"),
                    "vis_p50": rec.get("vis_p50"),
                    "vis_p90": rec.get("vis_p90"),
                    "nan_rate_xyz": rec.get("nan_rate_xyz"),
                    "wall_s": rec.get("wall_s"),
                    "eff_fps": rec.get("eff_fps"),
                    "mem_mb_delta": rec.get("mem_mb_delta"),
                    "rss_peak_mb": rec.get("rss_peak_mb"),
                    "cpu_util_pct": rec.get("cpu_util_pct"),
                    "infer_lat_median_ms": rec.get("infer_lat_median_ms"),
                    "infer_lat_p90_ms": rec.get("infer_lat_p90_ms"),
                    "infer_lat_p99_ms": rec.get("infer_lat_p99_ms"),
                }
            )

            if DEBUG and rec.get("type") == "video_ok":
                print(f"Processed video OK: {video_path}")
            if DEBUG and rec.get("type") == "video_error":
                print(f"Error processing video: {video_path} | {rec.get('error')}")

        except Exception as e:
            tb = traceback.format_exc(limit=5)
            err_rec = {
                "type": "video_error",
                "source": SOURCE_NAME,
                "ts_utc": now_iso(),
                "run_id": RUN_ID,
                "model": "openpose",
                "model_variant": MODEL_VARIANT,
                "video_path": video_path,
                "label": label,
                "error": str(e),
                "traceback": tb,
            }
            append_jsonl(RUN_LOG, err_rec)

            if INGEST_URL and token is not None:
                try:
                    send_request_api(INGEST_URL, err_rec, token)
                except Exception as api_e:
                    append_jsonl(
                        RUN_LOG,
                        {
                            "type": "api_error",
                            "ts_utc": now_iso(),
                            "video_path": video_path,
                            "label": label,
                            "error": str(api_e),
                        },
                    )

            meta_rows.append({"out_dir": "", "label": label, "video_path": video_path, "error": str(e)})

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(METRICS_DIR / "pose_meta.csv", index=False)
    meta.to_csv(RUN_SUMMARY, index=False)

    append_jsonl(RUN_LOG, {"type": "run_end", "ts_utc": now_iso(), "summary_csv": str(RUN_SUMMARY)})

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

    print("=== RUN SUMMARY (OpenPose; aligned schema) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Wrote pose_meta.csv and {RUN_LOG} and {RUN_SUMMARY}")


if __name__ == "__main__":
    main()
