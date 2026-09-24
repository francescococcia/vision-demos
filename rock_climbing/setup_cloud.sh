#!/usr/bin/env bash
# One-shot setup for a Linux cloud session (e.g. Claude Code from the phone).
#   bash rock_climbing/setup_cloud.sh
# Needs VLMRUN_API_KEY set as an environment variable in the session settings.
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null; then
  (sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg) \
    || (apt-get update -qq && apt-get install -y -qq ffmpeg)
fi
python3 -m pip install -q -r requirements.txt

ffmpeg -version | head -1
python3 -c "import cv2, numpy, scipy, openai; print('python deps ok')"
if [ -n "$VLMRUN_API_KEY" ]; then echo "VLMRUN_API_KEY set"; else echo "WARNING: VLMRUN_API_KEY not set"; fi
