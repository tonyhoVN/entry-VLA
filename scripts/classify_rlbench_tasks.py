#!/usr/bin/env python3
"""Classify single-arm RLBench tasks by demo horizon (run in the peract2 env, CoppeliaSim).

RLBench has no horizon labels. Each task's scripted demo visits the waypoint dummies
``waypoint0..N`` stored in its .ttm; their extension strings carry gripper commands
(``close_gripper()`` = a grasp), and some tasks re-run the waypoint list
(``register_waypoints_should_repeat``) for several objects. One CoppeliaSim session imports each
task model, counts these, and removes it again. No demo is executed.

Classes (thresholds are choices, recorded in the output):
  short_single : no repeat, <= 1 grasp, <= SHORT_MAX_WP waypoints
  long_horizon : repeat registered, or >= 2 grasps, or >= LONG_MIN_WP waypoints
  medium       : everything else

  python scripts/classify_rlbench_tasks.py --out docs/rlbench_task_horizon.json
"""

import argparse
import json
import os
import re

SHORT_MAX_WP, LONG_MIN_WP = 5, 8


def classify(r):
    if r["repeats"] or r["grasps"] >= 2 or r["waypoints"] >= LONG_MIN_WP:
        return "long_horizon"
    if r["grasps"] <= 1 and r["waypoints"] <= SHORT_MAX_WP:
        return "short_single"
    return "medium"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rlbench", default="ext/RLBench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()

    from pyrep import PyRep
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.object import Object

    tdir = os.path.join(a.rlbench, "rlbench", "tasks")
    names = a.tasks or sorted(f[:-3] for f in os.listdir(tdir) if f.endswith(".py") and f != "__init__.py")
    pr = PyRep()
    pr.launch(os.path.join(a.rlbench, "rlbench", "task_design.ttt"), headless=True)
    pr.start()
    rows = []
    try:
        for name in names:
            src = open(os.path.join(tdir, name + ".py")).read()
            r = {"task": name, "repeats": "register_waypoints_should_repeat" in src,
                 "waypoint_callbacks": len(re.findall(r"register_waypoint_ability_(start|end)", src))}
            try:
                base = pr.import_model(os.path.join(a.rlbench, "rlbench", "task_ttms", name + ".ttm"))
                wps, grasps, releases, ext_all = 0, 0, 0, []
                i = 0
                while Object.exists(f"waypoint{i}"):
                    ext = Dummy(f"waypoint{i}").get_extension_string() if Object.get_object_type(
                        f"waypoint{i}").name == "DUMMY" else ""
                    grasps += ext.count("close_gripper(")
                    releases += ext.count("open_gripper(")
                    ext_all.append(ext)
                    wps, i = wps + 1, i + 1
                r.update(waypoints=wps, grasps=grasps, releases=releases, extensions=ext_all)
                base.remove()
            except Exception as e:  # noqa: BLE001 - record and continue
                r.update(waypoints=None, grasps=None, releases=None, error=f"{type(e).__name__}: {e}")
            r["class"] = classify(r) if r.get("waypoints") is not None else "unknown"
            rows.append(r)
            print(f"{name:45s} wp={r['waypoints']} grasps={r['grasps']} repeat={r['repeats']} -> {r['class']}",
                  flush=True)
    finally:
        pr.stop()
        pr.shutdown()
    out = {"criteria": {"short_single": f"no repeat, <=1 grasp, <={SHORT_MAX_WP} waypoints",
                        "long_horizon": f"repeat, or >=2 grasps, or >={LONG_MIN_WP} waypoints",
                        "medium": "otherwise",
                        "note": "counts from the task .ttm waypoint dummies; dynamic waypoints added in code are not seen"},
           "tasks": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
