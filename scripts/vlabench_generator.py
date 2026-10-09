#!/usr/bin/env python3
"""Generate ONE VLABench scripted demonstration with the geometry needed for contact/phase analysis
(run in the ``vlabench`` env; ``ext/VLABench`` installed editable, assets downloaded).

VLABench's own ``scripts/trajectory_generation.py`` drops camera intrinsics/extrinsics and the
segmentation, and saves point clouds instead. This script runs the same expert skill sequence
(``env.get_expert_skill_sequence()``, keypoint + motion-planning skills, NOT teleoperation) and
records, for every observation the skills return:

  rgb        (T, C, H, W, 3) uint8         depth (T, C, H, W) float32, metres (MuJoCo/dm_control)
  seg_geom   (T, C, H, W) int32  MuJoCo geom id per pixel, -1 = no geom (privileged)
  cam_K      (T, C, 3, 3)                  cam_T_WC (T, C, 4, 4) camera->world, OpenCV (x right, y down, z fwd)
  tcp_T_W    (T, 4, 4) end_effector site -> world      finger_qpos (T, 2) finger joint positions (m)
  q_state / q_velocity / ee_state (VLABench, ee quat wxyz; its open flag is inverted upstream, unused)
  sim_finger_contact (T,) int  #MuJoCo contacts between finger geoms and non-robot geoms (privileged)
  sim_robot_contact  (T,) int  #MuJoCo contacts between any robot geom and non-robot geoms (privileged)
  skill_end  (S,) int  index of the last frame of each expert skill
  trajectory (N, 3+4+2?) VLABench waypoints in robot-base frame (as upstream)
attrs: format, task, seed, index, instruction, success, target_entity, entities, episode_config (json),
camera_names, robot_geom_ids, hand_geom_ids, finger_geom_ids, image size.

Output: <save_path>/<task>/seed<S>/episodes/episode<N>/data.hdf5 (+ demo.mp4 front-view preview).
The episode seed is ``seed * 1000 + index``. Failed demos are not saved (exit code 2).

  MUJOCO_GL=egl python scripts/vlabench_generator.py --task select_fruit --index 0 --seed 42 \
      --save_path data/raw/vlabench
"""

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

FORMAT = "entry_vla.vlabench_raw/1"
MJ_OBJ_GEOM = 5  # mjtObj.mjOBJ_GEOM


def geom_ids(physics, mjcf_root, body=None):
    root = mjcf_root.find("body", body) if body else mjcf_root
    return sorted(int(i) for i in physics.bind(root.find_all("geom")).element_id)


def camera_T_WC(physics, cam):
    """camera->world in OpenCV convention. MuJoCo cameras look along -z with y up; flip y and z."""
    T = np.eye(4)
    T[:3, :3] = physics.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = physics.data.cam_xpos[cam]
    return T


def camera_K(physics, cam, w, h):
    f = h / (2 * np.tan(np.radians(physics.model.cam_fovy[cam]) / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def contact_counts(physics, robot_ids, finger_ids):
    rs, fs = set(robot_ids), set(finger_ids)
    n_f = n_r = 0
    for c in physics.data.contact[:physics.data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in rs) == (b in rs):
            continue
        n_r += 1
        n_f += (a in fs) or (b in fs)
    return n_f, n_r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="registered VLABench task, e.g. select_fruit")
    ap.add_argument("--index", type=int, default=0, help="episode index")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    ep_seed = a.seed * 1000 + a.index
    random.seed(ep_seed)
    np.random.seed(ep_seed)
    import VLABench.robots  # noqa: F401 - registers robots
    import VLABench.tasks  # noqa: F401 - registers tasks
    from VLABench.envs import load_env

    t0 = time.time()
    env = load_env(a.task, robot=a.robot)
    env.reset()
    episode_config = env.save()
    physics = env.physics
    w, h = env.render_options["width"], env.render_options["height"]
    ncam = physics.model.ncam
    cam_names = [physics.model.id2name(c, "camera") for c in range(ncam)]
    robot_model = env.robot.mjcf_model
    robot_ids = geom_ids(physics, robot_model)
    hand_ids = geom_ids(physics, robot_model, "hand")
    finger_ids = sorted(set(geom_ids(physics, robot_model, "left_finger")) |
                        set(geom_ids(physics, robot_model, "right_finger")))
    fj = [robot_model.find("joint", n) for n in ("finger_joint1", "finger_joint2")]
    site = env.robot.end_effector_site

    orig = env.get_observation

    def get_observation(*args, **kw):
        o = orig(*args, **kw)
        seg = o["segmentation"]
        o["seg_geom"] = np.where(seg[..., 1] == MJ_OBJ_GEOM, seg[..., 0], -1).astype(np.int32)
        o["cam_K"] = np.stack([camera_K(physics, c, w, h) for c in range(ncam)])
        o["cam_T_WC"] = np.stack([camera_T_WC(physics, c) for c in range(ncam)])
        T = np.eye(4)
        T[:3, :3] = np.asarray(physics.bind(site).xmat).reshape(3, 3)
        T[:3, 3] = physics.bind(site).xpos
        o["tcp_T_W"] = T
        o["finger_qpos"] = np.array([float(physics.bind(j).qpos[0]) for j in fj])
        o["sim_finger_contact"], o["sim_robot_contact"] = contact_counts(physics, robot_ids, finger_ids)
        return o

    env.get_observation = get_observation
    instruction = env.task.get_instruction()
    target_entity = env.task.config_manager.target_entity
    skills = env.get_expert_skill_sequence()
    if skills is None:
        raise SystemExit(f"{a.task}: no expert skill sequence")

    obs, waypoints, skill_end, success = [], [], [], False
    for skill in skills:
        o, wp, stage_ok, success = skill(env)
        obs.extend(o)
        waypoints.extend(wp)
        skill_end.append(len(obs) - 1)
        if success:
            break
    print(f"{a.task} #{a.index}: {len(obs)} frames, {len(skill_end)} skills, success={success}, "
          f"{time.time() - t0:.0f}s", flush=True)
    if not success:
        env.close()
        sys.exit(2)

    import h5py
    out = os.path.join(a.save_path, a.task, f"seed{a.seed}", "episodes", f"episode{a.index}")
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "data.hdf5.tmp")
    robot_pos = np.asarray(env.robot.robot_config["position"], dtype=np.float64)
    traj = [np.asarray(p, dtype=np.float64) - np.concatenate([robot_pos, np.zeros(len(p) - 3)]) for p in waypoints]
    with h5py.File(tmp, "w") as f:
        gz = dict(compression="gzip", compression_opts=4)
        img_chunk = (1, 1, h, w)   # one chunk per frame and camera: cheap random frame access
        f.create_dataset("rgb", data=np.stack([o["rgb"] for o in obs]).astype(np.uint8), chunks=img_chunk + (3,), **gz)
        f.create_dataset("depth", data=np.stack([o["depth"] for o in obs]).astype(np.float32), chunks=img_chunk, **gz)
        f.create_dataset("seg_geom", data=np.stack([o["seg_geom"] for o in obs]), chunks=img_chunk, **gz)
        for k in ("cam_K", "cam_T_WC", "tcp_T_W", "finger_qpos", "q_state", "q_velocity", "ee_state"):
            f.create_dataset(k, data=np.stack([np.asarray(o[k], dtype=np.float64) for o in obs]))
        for k in ("sim_finger_contact", "sim_robot_contact"):
            f.create_dataset(k, data=np.array([o[k] for o in obs], dtype=np.int32))
        f.create_dataset("skill_end", data=np.array(skill_end, dtype=np.int32))
        if traj:
            f.create_dataset("trajectory", data=np.stack(traj).astype(np.float32))
        f.attrs.update(format=FORMAT, task=a.task, seed=a.seed, index=a.index, episode_seed=ep_seed,
                       instruction=str(instruction), success=bool(success), target_entity=str(target_entity),
                       entities=json.dumps(list(env.task.entities.keys())),
                       episode_config=json.dumps(episode_config, default=str), camera_names=json.dumps(cam_names),
                       robot_geom_ids=robot_ids, hand_geom_ids=hand_ids, finger_geom_ids=finger_ids,
                       image_size=[w, h], depth_unit="m", quat_order_ee_state="wxyz",
                       provenance="VLABench scripted expert skills (keypoints + motion planning), not teleoperation")
    os.replace(tmp, os.path.join(out, "data.hdf5"))
    if not a.no_video:
        import mediapy
        mediapy.write_video(os.path.join(out, "demo.mp4"),
                            [np.vstack([np.hstack(o["rgb"][:2]), np.hstack(o["rgb"][2:4])]) if ncam >= 4
                             else np.hstack(o["rgb"]) for o in obs], fps=10)
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
