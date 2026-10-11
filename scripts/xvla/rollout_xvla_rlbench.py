#!/usr/bin/env python3
"""Two-phase RLBench rollout: motion planning to the entry pose, then the fine-tuned X-VLA policy
(run in the `peract2` env with a display; the policy runs in scripts/xvla/xvla_policy_server.py).

Scene: restored from a recorded demo (``Demo.random_seed`` -> ``reset_to_demo``), e.g. a seed that was
never used for training. Entry pose: the demo TCP pose at ``split_frame`` of its contact split
(contact/contact.json; ORACLE entry from the demo, not a learned predictor) and the demo gripper state.
  Phase 1  planner to the entry pose, collision-checked, executed step by step; attempts in order:
           PyRep get_path -> SBL with 10 goal configs -> via a pre-entry point back along the approach
           axis + linear -> (fallback) replay of the demo's own planned joint path to the entry frame.
           The method used is in rollout.json and the video label; gripper set to the demo state.
  Phase 2  X-VLA: query with front/overhead/wrist RGB + instruction + current TCP pose; execute the
           first --exec-steps absolute EE6D targets of each chunk with Jacobian IK (joint targets,
           stepped until reached or stalled), gripper from p(closed) > 0.5; repeat up to --max-queries.
Every simulation step is recorded (front | overhead | wrist, phase label) to <out>/rollout.mp4;
<out>/rollout.json has the success flag, phase lengths, IK failures and latencies.

  DISPLAY=:1 python scripts/xvla/rollout_xvla_rlbench.py --task close_microwave \
      --episode data/raw/rlbench_single/close_microwave/variation0/seed1001/episodes/episode0 \
      --contact data/entry_process/xvla_rollout/close_microwave/seed1001/episode0/contact/contact.json \
      --port 6010 --out runs/xvla_close_microwave/rollouts/seed1001
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time

import numpy as np

VIEWS = ["front", "overhead", "wrist"]


def quat_xyzw_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot6d_xvla(R):
    return R[:, :2].reshape(6)   # X-VLA layout (row-major R[:, :2])


def rot6d_xvla_to_R(v):
    a1, a2 = v[0:5:2], v[1:6:2]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - b1.dot(a2) * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


class Writer:
    def __init__(self, path, fps, ffmpeg):
        self.path, self.fps, self.ffmpeg, self.p = path, fps, ffmpeg, None

    def add(self, frame):
        if self.p is None:
            h, w = frame.shape[:2]
            self.p = subprocess.Popen([self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
                                       "-pix_fmt", "yuv420p", self.path], stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", required=True, help="recorded demo dir (low_dim_obs.pkl, variation_*.pkl)")
    ap.add_argument("--contact", required=True, help="contact.json of that demo (split_frame = entry)")
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--authkey", default="entry-vla-xvla")
    ap.add_argument("--exec-steps", type=int, default=10, help="targets executed per X-VLA chunk")
    ap.add_argument("--max-queries", type=int, default=40)
    ap.add_argument("--max-ik-steps", type=int, default=20, help="sim steps per target before moving on")
    ap.add_argument("--pre-entry-offset", type=float, default=0.08, help="via point back along the approach axis (m)")
    ap.add_argument("--no-demo-path-fallback", dest="demo_path_fallback", action="store_false",
                    help="do not replay the demo's planned joint path when every planning attempt fails")
    ap.add_argument("--image-size", type=int, default=256, help="must match the training images")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or os.path.expanduser("~/miniconda3/envs/entryproc/bin/ffmpeg"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    from multiprocessing.connection import Client
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench import ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment
    from rlbench.observation_config import CameraConfig
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from rlbench_single_generator import JointVelocityCompat, _patch_unimanual_release  # noqa: F401

    contact = json.load(open(a.contact))
    entry = contact["split_frame"]
    if entry is None:
        raise SystemExit("contact split has no split_frame")
    demo = pickle.load(open(os.path.join(a.episode, "low_dim_obs.pkl"), "rb"))
    variation = pickle.load(open(os.path.join(a.episode, "variation_number.pkl"), "rb"))
    instruction = pickle.load(open(os.path.join(a.episode, "variation_descriptions.pkl"), "rb"))[0]
    e_obs = demo[entry]
    entry_pose = np.asarray(e_obs.gripper_pose, dtype=np.float64)

    _patch_unimanual_release()
    oc = ObservationConfig()
    oc.set_all_low_dim(True)
    oc.camera_configs = {c: CameraConfig(image_size=(a.image_size, a.image_size), depth=False, point_cloud=False,
                                         mask=False, rgb=True) for c in VIEWS}
    env = Environment(action_mode=MoveArmThenGripper(JointVelocityCompat(), Discrete()), obs_config=oc,
                      robot_setup="panda", headless=True)
    env.launch()
    writer = Writer(os.path.join(a.out, "rollout.mp4"), a.fps, a.ffmpeg)
    rec = {"task": a.task, "episode": os.path.abspath(a.episode), "instruction": instruction, "entry_frame": entry,
           "entry_source": "demo TCP pose at contact split_frame (oracle)", "views": VIEWS}
    try:
        task_env = env.get_task(task_file_to_task_class(a.task, False))
        task_env.set_variation(variation)
        task_env.reset_to_demo(demo)
        scene = task_env._scene
        robot = scene.robot
        # the generator's joint-velocity action mode disables the arm position controller; planner paths
        # and IK both drive joint POSITION targets, so enable it (as RLBench's planning/IK modes do)
        robot.arm.set_control_loop_enabled(True)
        obs0 = scene.get_observation()
        rec["restore_check"] = {"gripper_pose_diff_m": float(np.linalg.norm(obs0.gripper_pose[:3] - demo[0].gripper_pose[:3])),
                                "task_state_diff": float(np.abs(np.asarray(obs0.task_low_dim_state).ravel()[:10] -
                                                                np.asarray(demo[0].task_low_dim_state).ravel()[:10]).max())}
        state = {"phase": "phase 1: motion planning", "success": False, "step": 0}

        def frame(obs):
            tiles = [obs.perception_data[f"{v}_rgb"] for v in VIEWS]
            img = np.ascontiguousarray(np.hstack(tiles))
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            color = (60, 140, 255) if state["phase"].startswith("phase 1") else (255, 140, 40)
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
            txt = f"{state['phase']} | step {state['step']}" + (" | SUCCESS" if state["success"] else "")
            cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, instruction, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return img

        def on_step():
            state["step"] += 1
            if not state["success"]:
                state["success"] = bool(scene.task.success()[0])
            writer.add(frame(scene.get_observation()))

        scene.register_step_callback(on_step)

        def set_gripper(open_amount):
            target = 1.0 if open_amount > 0.5 else 0.0
            if (all(x > 0.9 for x in robot.gripper.get_open_amount())) == (target == 1.0):
                return
            done = False
            for _ in range(200):
                if done:
                    break
                done = robot.gripper.actuate(target, velocity=0.2)
                scene.step()
            if target == 0.0:
                for g in scene.task.get_graspable_objects():
                    robot.gripper.grasp(g)
            else:
                robot.gripper.release()

        # ---- phase 1: motion planning to the entry pose
        t0 = time.time()
        writer.add(frame(obs0))
        def run_path(path):
            done = False
            while not done:
                done = path.step()
                scene.step()

        def plan_attempts():
            pos, quat = entry_pose[:3], entry_pose[3:7]
            yield "get_path (linear, else SBL)", lambda: run_path(robot.arm.get_path(pos, quaternion=quat, trials=300))
            yield "SBL, 10 goal configs", lambda: run_path(robot.arm.get_nonlinear_path(
                pos, quaternion=quat, trials=1000, max_configs=10, distance_threshold=1.5, max_time_ms=50,
                trials_per_goal=5))
            approach = quat_xyzw_to_R(quat)[:, 2]          # TCP z = approach axis
            pre = pos - a.pre_entry_offset * approach

            def via():
                run_path(robot.arm.get_nonlinear_path(pre, quaternion=quat, trials=1000, max_configs=10,
                                                      distance_threshold=1.5, max_time_ms=50, trials_per_goal=5))
                run_path(robot.arm.get_linear_path(pos, quaternion=quat))
            yield f"via pre-entry {a.pre_entry_offset:.2f} m back, then linear", via
            if a.demo_path_fallback:
                def replay():
                    for o in demo[1:entry + 1]:
                        q = np.asarray(o.joint_positions, dtype=np.float64)
                        robot.arm.set_joint_target_positions(q)
                        for _ in range(a.max_ik_steps):
                            scene.step()
                            if np.allclose(robot.arm.get_joint_positions(), q, atol=0.01):
                                break
                yield "replay of the demo's planned joint path (RLBench demo planner)", replay

        plan_ok, attempts = False, []
        for name, fn in plan_attempts():
            state["phase"] = f"phase 1: motion planning [{name.split(' (')[0]}]"
            try:
                fn()
                plan_ok = True
                attempts.append({"method": name, "ok": True})
                break
            except ConfigurationPathError as e:
                attempts.append({"method": name, "ok": False, "error": str(e)})
        rec["plan_attempts"] = attempts
        set_gripper(float(e_obs.gripper_open))
        reached = np.asarray(scene.get_observation().gripper_pose)
        rec["phase1"] = {"planner": attempts[-1]["method"] if attempts else None, "ok": plan_ok, "sim_steps": state["step"],
                         "entry_pos_error_m": float(np.linalg.norm(reached[:3] - entry_pose[:3])),
                         "time_s": round(time.time() - t0, 1)}
        print("phase 1", rec["phase1"], flush=True)

        # ---- phase 2: X-VLA closed loop
        state["phase"] = "phase 2: X-VLA"
        p1_steps = state["step"]
        conn = Client(("localhost", a.port), authkey=a.authkey.encode())
        lat, ik_fail, queries = [], 0, 0
        try:
            while queries < a.max_queries and not state["success"]:
                obs = scene.get_observation()
                gp = np.asarray(obs.gripper_pose, dtype=np.float64)
                left = np.concatenate([gp[:3], rot6d_xvla(quat_xyzw_to_R(gp[3:7])), [1.0 - float(obs.gripper_open)]])
                conn.send({"images": [obs.perception_data[f"{v}_rgb"] for v in VIEWS],
                           "proprio": np.concatenate([left, np.zeros(10)]).astype(np.float32),
                           "instruction": instruction})
                rep = conn.recv()
                lat.append(rep["latency_s"])
                queries += 1
                for act in rep["actions"][:a.exec_steps]:
                    pos, quat = act[:3].astype(np.float64), R_to_quat_xyzw(rot6d_xvla_to_R(act[3:9].astype(np.float64)))
                    try:
                        q = robot.arm.solve_ik_via_jacobian(pos, quaternion=quat)
                        robot.arm.set_joint_target_positions(q)
                        prev = None
                        for _ in range(a.max_ik_steps):
                            scene.step()
                            cur = np.asarray(robot.arm.get_joint_positions())
                            if np.allclose(cur, q, atol=0.01) or (prev is not None and np.allclose(cur, prev, atol=0.001)):
                                break
                            prev = cur
                    except IKError:
                        ik_fail += 1
                    set_gripper(0.0 if act[9] > 0.5 else 1.0)
                    if state["success"]:
                        break
        finally:
            conn.close()
        rec["phase2"] = {"policy": "X-VLA (server)", "queries": queries, "exec_steps_per_chunk": a.exec_steps,
                         "sim_steps": state["step"] - p1_steps, "ik_failures": ik_fail,
                         "latency_s_mean": float(np.mean(lat)) if lat else None}
        rec["success"] = state["success"]
        # hold the last frame for a second
        last = frame(scene.get_observation())
        for _ in range(a.fps):
            writer.add(last)
    finally:
        writer.close()
        json.dump(rec, open(os.path.join(a.out, "rollout.json"), "w"), indent=1)
        env.shutdown()
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
