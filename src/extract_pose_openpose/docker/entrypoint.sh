#!/bin/sh
set -e

python3 -u make_manifest.py
python3 -u extract_pose_openpose.py
