#!/usr/bin/env python3
"""Visualise the frame-0 object detection used by the entry model (Qwen3-VL box + SAM 2.1 mask).

One row per camera: RGB | box + SAM mask overlay | final fused object cloud (after the multi-view check
and table-top cluster) projected back into that camera. Optional: predicted entry (--entry-json, top
candidate, red) and demo oracle entry (--contact, green) projected as dots.

  python scripts/visualize_detection.py --episode <raw episode> --out det.png [--entry-json entry_prediction.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entry-json", default=None)
    ap.add_argument("--contact", default=None)
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--cache-index", default=None, help="index.json of the training cache: use its detector settings")
    a = ap.parse_args()
    import cv2
    from entry_vla.data.entry_diffusion_data import detect_object_frame0
    from entry_vla.data.peract2_raw import demo_observations, load_pickle
    from entry_vla.geometry.camera import project_world
    from entry_vla.perception.grounding import Qwen3VLGrounder
    from entry_vla.perception.segmentation import make_segmenter

    instr = load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))[0]
    dbg = {}
    det = json.load(open(a.cache_index)).get("detector", {}) if a.cache_index else {}
    kw = {k: det[k] for k in ("cloud_cams", "crop_cams", "cloud_filter", "voxel_m") if isinstance(det, dict) and k in det}
    s = detect_object_frame0(a.episode, instr, Qwen3VLGrounder(a.vlm, device=a.device), make_segmenter("sam2.1", a.device),
                             debug=dbg, **kw)
    marks = []
    if a.entry_json:
        c = json.load(open(a.entry_json))["candidates"][0]
        marks.append((np.array(c["pos_world"]), (255, 40, 40), "predicted entry"))
    if a.contact:
        j = json.load(open(a.contact))
        obs = demo_observations(load_pickle(os.path.join(a.episode, "low_dim_obs.pkl")))
        marks.append((np.asarray(obs[j["split_frame"]].gripper_pose[:3]), (40, 220, 40), "demo entry"))
    P = dbg.get("object_points_W", np.zeros((0, 3)))
    rows = []
    for cam, v in dbg.get("views", {}).items():
        rgb, m, box = v["rgb"], v["mask"], v["box"]
        ov = rgb.copy()
        ov[m] = (0.45 * ov[m] + 0.55 * np.array([255, 140, 0])).astype(np.uint8)
        cv2.rectangle(ov, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (40, 120, 255), 1)
        pc = (rgb * 0.5).astype(np.uint8)
        if len(P):
            uv, z = project_world(P, v["K"], v["T_WC"])
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < rgb.shape[1]) & (uv[:, 1] < rgb.shape[0])
            pc[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = (0, 255, 255)
        for p, col, _ in marks:
            uv1, z1 = project_world(p[None], v["K"], v["T_WC"])
            if z1[0] > 0:
                for im in (ov, pc):
                    cv2.circle(im, (int(uv1[0, 0]), int(uv1[0, 1])), 4, col, -1)
        row = np.hstack([rgb, ov, pc])
        row = cv2.resize(row, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(row, f"{cam}: RGB | Qwen box + SAM mask | fused object cloud ({len(P)} pts)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(row)
    img = np.vstack(rows)
    legend = f"'{instr}'" + "".join(f"  |  {name}: {'red' if col[0] > 200 else 'green'} dot" for _, col, name in marks)
    bar = np.full((34, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, legend, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, np.vstack([bar, img])[..., ::-1])
    print(a.out, "detected" if s is not None else "NOT detected")


if __name__ == "__main__":
    main()
