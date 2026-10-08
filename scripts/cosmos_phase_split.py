#!/usr/bin/env python3
"""Ask Cosmos 3 (Reasoner, served by vLLM with an OpenAI-compatible API) to split one
RLBench episode into alternating transition / manipulation segments (any number of
skills), with its reasoning. The prompt is deliberately generic: no cues about stops,
gripper state or contact, so the split comes from the model's own video reasoning.

Steps:
  1. ``--make-video`` builds a label-free 4 fps clip (the rate the Cosmos 3 model card
     recommends) from the raw episode: front | overhead | wrist RGB side by side, with only
     the source frame index burned in for humans; the model answers in seconds, which are
     mapped back to source frames through clip.json.
     No phase labels or detector output are drawn (that would leak the answer).
  2. ``--query`` sends the clip (base64 data URL, so it works through an SSH tunnel) and the
     prompt, then writes the raw answer and the parsed JSON to ``--out``.
     If ``transition.json`` from scripts/detect_transition_rlbench.py exists in ``--out``,
     the parsed frame is compared with the rule-based entry frame.

Only the standard library plus numpy/PIL/cv2/ffmpeg are needed (no openai package).

  python scripts/cosmos_phase_split.py --make-video --query \
      --episode data/raw/rlbench_single/push_button/variation4/seed_unknown/episodes/episode0 \
      --out data/entry_process/push_button/variation4/episode0 \
      --base-url http://localhost:8000/v1
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.peract2_raw import load_pickle  # noqa: E402

VIEW_CAMERAS = ["front", "overhead", "wrist"]
RECORD_HZ = 20  # RLBench demo frames (0.05 s step, assumed)

PROMPT = """The video shows a robot performing the task: "{instruction}".
Panels, left to right: front camera, overhead camera, wrist camera. The video is {duration:.2f} seconds long.

Split the whole video into consecutive segments of two kinds:
- transition: the robot moves to get ready for the next skill, without interacting with objects;
- manipulation: the robot performs a skill on an object.
A task may contain one or several skills, so there may be several transition and manipulation segments.

Answer in the following format:
<think>
your reasoning, citing timestamps in seconds
</think>

<answer>
{{"segments": [{{"phase": "transition" or "manipulation", "skill": "short skill name, or null for transition", "start_s": float, "end_s": float, "explanation": "why this segment starts and ends here"}}]}}
</answer>"""


def make_video(ep, out_path, fps, size):
    import cv2
    from PIL import Image
    n = len([f for f in os.listdir(os.path.join(ep, "front_rgb")) if f.endswith(".png")])
    step = max(1, round(RECORD_HZ / fps))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)  # keep the final state
    frames = []
    for i in idx:
        tiles = []
        for cam in VIEW_CAMERAS:
            im = np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
            tiles.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
        img = np.ascontiguousarray(np.concatenate(tiles, 1))
        cv2.rectangle(img, (0, 0), (130, 30), (0, 0, 0), -1)
        cv2.putText(img, f"frame {i:03d}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(img)
    h, w = frames[0].shape[:2]
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return {"video": out_path, "fps": fps, "source_frames": idx, "num_source_frames": n, "cameras": VIEW_CAMERAS}


def post(base_url, payload, timeout):
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def served_model(base_url, timeout=10):
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
        return json.load(r)["data"][0]["id"]


def parse_answer(text):
    """JSON inside <answer>...</answer>, else the last parseable {...} block in the text."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    cands = [m.group(1)] if m else []
    cands += [text[k.start():] for k in re.finditer(r"\{", text)]  # outermost object first
    dec = json.JSONDecoder()
    for c in cands:
        c = c.strip().strip("`").removeprefix("json").strip()
        try:
            return dec.raw_decode(c)[0]
        except json.JSONDecodeError:
            continue
    return None


def to_source_frame(t, clip, fps):
    """seconds in the clip -> clip frame -> source frame of the raw episode"""
    ci = min(max(int(round(float(t) * fps)), 0), len(clip["source_frames"]) - 1)
    return clip["source_frames"][ci]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--query", action="store_true")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--size", type=int, default=256, help="per-camera tile size in the clip")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="greedy (0) can loop in long <think> outputs; sampling avoids that")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--run-name", default="", help="suffix for response/result files, e.g. seed1")
    ap.add_argument("--dry-run", action="store_true", help="print the request without sending it")
    a = ap.parse_args()

    cdir = os.path.join(a.out, "cosmos")
    os.makedirs(cdir, exist_ok=True)
    video = os.path.join(cdir, f"episode_{a.fps}fps.mp4")
    clip_meta_path = os.path.join(cdir, "clip.json")
    if a.make_video:
        meta = make_video(a.episode, video, a.fps, a.size)
        with open(clip_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"clip: {video} ({len(meta['source_frames'])} frames at {a.fps} fps)")
    if not a.query:
        return
    if not os.path.exists(video):
        raise SystemExit("no clip yet; run with --make-video")

    descr = load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))
    with open(clip_meta_path) as f:
        clip = json.load(f)
    duration = len(clip["source_frames"]) / a.fps
    prompt = PROMPT.format(instruction=descr[0] if descr else "unknown", duration=duration, fps=a.fps)
    with open(video, "rb") as f:
        url = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
    payload = {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}},
                                                          {"type": "text", "text": prompt}]}],
               "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
               "repetition_penalty": a.repetition_penalty, "seed": a.seed}
    if a.dry_run:
        print(prompt)
        print(f"\nvideo payload: {len(url) / 1e6:.2f} MB base64, endpoint {a.base_url}")
        return

    payload["model"] = served_model(a.base_url)
    t0 = time.time()
    rep = post(a.base_url, payload, a.timeout)
    text = rep["choices"][0]["message"]["content"]
    parsed = parse_answer(text)
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    segments = []
    for seg in (parsed or {}).get("segments", []):
        try:
            segments.append({**seg, "start_frame": to_source_frame(seg["start_s"], clip, a.fps),
                             "end_frame": to_source_frame(seg["end_s"], clip, a.fps)})
        except (KeyError, TypeError, ValueError):
            segments.append({**seg, "start_frame": None, "end_frame": None})
    result = {"model": payload["model"], "endpoint": a.base_url, "latency_s": round(time.time() - t0, 2),
              "fps": a.fps, "sampling": {k: payload[k] for k in ("temperature", "top_p", "repetition_penalty", "seed", "max_tokens")},
              "finish_reason": rep["choices"][0].get("finish_reason"), "prompt": prompt,
              "reasoning": m.group(1).strip() if m else None, "segments": segments,
              "parse_ok": parsed is not None, "usage": rep.get("usage")}
    rule_path = os.path.join(a.out, "transition.json")
    manip = [g for g in segments if g.get("phase") == "manipulation" and g.get("start_frame") is not None]
    if manip and os.path.exists(rule_path):  # rule-based detector gives the first handoff only
        with open(rule_path) as f:
            rule = json.load(f)
        result["compare_rule_based"] = {
            "rule_entry_frame": rule["entry_frame"], "cosmos_first_manipulation_start_frame": manip[0]["start_frame"],
            "entry_frame_error": manip[0]["start_frame"] - rule["entry_frame"],
            "clip_frame_spacing": round(RECORD_HZ / a.fps)}
    with open(os.path.join(cdir, f"response{'_' + a.run_name if a.run_name else ''}.md"), "w") as f:
        f.write(text)
    with open(os.path.join(cdir, f"result{'_' + a.run_name if a.run_name else ''}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(text)
    print("\n" + json.dumps({k: result.get(k) for k in ("segments", "compare_rule_based", "finish_reason", "latency_s")}, indent=2))


if __name__ == "__main__":
    main()
