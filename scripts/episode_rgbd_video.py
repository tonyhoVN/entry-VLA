#!/usr/bin/env python3
"""Whole-episode video of every camera of one single-arm RLBench episode: top row RGB,
bottom row depth buffer (per-camera normalised for display, not metric), frame index in a
banner. Written next to the processed outputs; the raw episode stays untouched.

  python scripts/episode_rgbd_video.py --episode <raw episode dir> --out <dir>/episode_rgbd.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_transition_rlbench import tile, write_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--fps", type=int, default=20, help="playback rate (recording is 20 Hz)")
    ap.add_argument("--scale", type=int, default=1, help="upscale factor per camera tile")
    a = ap.parse_args()
    n = len([f for f in os.listdir(os.path.join(a.episode, "front_rgb")) if f.endswith(".png")])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_video(a.out, [tile(a.episode, i, "", a.scale) for i in range(n)], a.fps)
    print(f"{a.out}: {n} frames")


if __name__ == "__main__":
    main()
