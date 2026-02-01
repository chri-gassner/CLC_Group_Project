#!/bin/sh
set -e

python -u make_manifest.py
python -u extract_pose_mediapipe.py
